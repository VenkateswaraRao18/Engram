from __future__ import annotations

import json
import re
import uuid
import logging
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from ..models import ExtractionResult, Memory, Entity, Relation, MemoryType
from .base import LLMProvider
from .prompts import EXTRACTION_SYSTEM_PROMPT, format_messages

logger = logging.getLogger(__name__)


def _clean_json(text: str) -> str:
    """Remove JS-style comments and trailing commas so standard json.loads works."""
    text = re.sub(r"//[^\n]*", "", text)          # strip // comments
    text = re.sub(r",(\s*[}\]])", r"\1", text)    # strip trailing commas
    return text


class OllamaProvider:
    """Ollama-backed LLM provider."""

    def __init__(self, model: str = "llama3.1"):
        self.model = model

    def complete(self, system: str, user: str) -> str:
        import ollama

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0},
        )
        return response.message.content


class GeminiProvider:
    """Google Gemini-backed LLM provider via direct REST API."""

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        import os
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key required: pass api_key= or set GEMINI_API_KEY env var"
            )
        self._api_key = key
        self.model = model

    def complete(self, system: str, user: str) -> str:
        import json
        import time
        import urllib.error
        import urllib.request

        combined = f"{system}\n\n{user}"
        fallbacks = [self.model, "gemini-2.0-flash-lite", "gemini-1.5-flash"]
        last_err: Exception = RuntimeError("No models tried")

        for attempt, model in enumerate(fallbacks):
            url = f"{self._BASE_URL.format(model)}?key={self._api_key}"
            payload = json.dumps({
                "contents": [{"parts": [{"text": combined}]}],
            }).encode()
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode()
                except Exception:
                    body = ""
                logger.warning("Gemini HTTP %s for model %s: %s", e.code, model, body[:300])
                last_err = e
                if e.code in (429, 503):
                    time.sleep(2 ** attempt)
                continue
            except Exception as e:
                logger.warning("Gemini call error for model %s: %s", model, e)
                last_err = e
                continue

        raise RuntimeError(f"All Gemini models unavailable. Last: {last_err}")


class LLMExtractor:
    """Extracts memories, entities, and relations from conversation messages."""

    def __init__(self, llm: LLMProvider, window: int = 6):
        self.llm = llm
        self.window = window

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    def _call_llm(self, user_prompt: str) -> str:
        return self.llm.complete(EXTRACTION_SYSTEM_PROMPT, user_prompt)

    def extract(self, messages: list[dict], user_id: str) -> ExtractionResult:
        """Extract memories, entities, and relations from a conversation window."""
        if not messages:
            return ExtractionResult()

        # Use only the last `window` messages
        window_messages = messages[-self.window :]
        user_prompt = format_messages(window_messages)

        try:
            raw = self._call_llm(user_prompt)
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return ExtractionResult()

        try:
            # Strip markdown code fences if present
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                lines = [l for l in lines if not l.startswith("```")]
                text = "\n".join(lines).strip()

            text = _clean_json(text)
            data = json.loads(text)
        except Exception as exc:
            logger.warning("JSON parse failed: %s\nRaw: %s", exc, raw)
            return ExtractionResult()

        try:
            return self._build_extraction_result(data, user_id)
        except Exception as exc:
            logger.warning("Building extraction result failed: %s", exc)
            return ExtractionResult()

    def _build_extraction_result(self, data: dict, user_id: str) -> ExtractionResult:
        """Build ExtractionResult from parsed JSON data."""
        raw_memories = data.get("memories", [])
        raw_entities = data.get("entities", [])
        raw_relations = data.get("relations", [])

        # Build Memory objects
        memories: list[Memory] = []
        for rm in raw_memories:
            try:
                mt = MemoryType(rm.get("memory_type", "semantic"))
            except ValueError:
                mt = MemoryType.SEMANTIC
            mem = Memory(
                user_id=user_id,
                content=rm["content"],
                memory_type=mt,
                importance=float(rm.get("importance", 0.5)),
            )
            memories.append(mem)

        # Build Entity objects; track by lower-case name for relation lookup
        entities: list[Entity] = []
        entity_name_to_obj: dict[str, Entity] = {}
        for re in raw_entities:
            ent = Entity(
                user_id=user_id,
                name=re["name"],
                entity_type=re.get("entity_type", "OTHER"),
                aliases=re.get("aliases", []),
            )
            entities.append(ent)
            entity_name_to_obj[ent.name.lower()] = ent

        # Build Relation objects
        relations: list[Relation] = []
        fallback_memory_id = memories[0].id if memories else str(uuid.uuid4())
        for rr in raw_relations:
            src_name = rr.get("source_entity_name", "").lower()
            tgt_name = rr.get("target_entity_name", "").lower()
            src_entity = entity_name_to_obj.get(src_name)
            tgt_entity = entity_name_to_obj.get(tgt_name)
            if src_entity is None or tgt_entity is None:
                continue

            # Link this relation to the memory whose content mentions the target
            # entity. This ensures supersession marks the correct memory as stale
            # rather than whichever memory happened to be first in the list.
            memory_id = fallback_memory_id
            for mem in memories:
                if tgt_name and tgt_name in mem.content.lower():
                    memory_id = mem.id
                    break

            rel = Relation(
                user_id=user_id,
                source_entity_id=src_entity.id,
                relation_type=rr.get("relation_type", "OTHER"),
                target_entity_id=tgt_entity.id,
                memory_id=memory_id,
                confidence=float(rr.get("confidence", 0.8)),
            )
            relations.append(rel)

        return ExtractionResult(memories=memories, entities=entities, relations=relations)
