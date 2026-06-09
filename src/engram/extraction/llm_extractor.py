from __future__ import annotations

import json
import uuid
import logging
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from ..models import ExtractionResult, Memory, Entity, Relation, MemoryType
from .base import LLMProvider
from .prompts import EXTRACTION_SYSTEM_PROMPT, format_messages

logger = logging.getLogger(__name__)


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
                # Remove first and last fence lines
                lines = [l for l in lines if not l.startswith("```")]
                text = "\n".join(lines).strip()

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
        first_memory_id = memories[0].id if memories else str(uuid.uuid4())
        for rr in raw_relations:
            src_name = rr.get("source_entity_name", "").lower()
            tgt_name = rr.get("target_entity_name", "").lower()
            src_entity = entity_name_to_obj.get(src_name)
            tgt_entity = entity_name_to_obj.get(tgt_name)
            if src_entity is None or tgt_entity is None:
                continue
            rel = Relation(
                user_id=user_id,
                source_entity_id=src_entity.id,
                relation_type=rr.get("relation_type", "OTHER"),
                target_entity_id=tgt_entity.id,
                memory_id=first_memory_id,
                confidence=float(rr.get("confidence", 0.8)),
            )
            relations.append(rel)

        return ExtractionResult(memories=memories, entities=entities, relations=relations)
