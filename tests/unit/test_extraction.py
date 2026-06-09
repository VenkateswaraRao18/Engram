from __future__ import annotations

import json

import pytest

from engram.extraction.llm_extractor import LLMExtractor
from engram.models import ExtractionResult


def test_extraction_basic(fake_llm):
    extractor = LLMExtractor(llm=fake_llm)
    messages = [
        {"role": "user", "content": "Hi, I'm Alex. I live in Tampa."},
        {"role": "assistant", "content": "Hello Alex! Nice to meet you."},
    ]
    result = extractor.extract(messages, user_id="u1")

    assert len(result.memories) == 1
    assert result.memories[0].content == "User lives in Tampa"
    assert len(result.entities) == 2
    assert len(result.relations) == 1


def test_extraction_entity_ids_match_relation(fake_llm):
    extractor = LLMExtractor(llm=fake_llm)
    messages = [{"role": "user", "content": "I live in Tampa."}]
    result = extractor.extract(messages, user_id="u1")

    entity_ids = {e.id for e in result.entities}
    for rel in result.relations:
        assert rel.source_entity_id in entity_ids
        assert rel.target_entity_id in entity_ids


def test_extraction_relation_memory_id(fake_llm):
    extractor = LLMExtractor(llm=fake_llm)
    messages = [{"role": "user", "content": "I live in Tampa."}]
    result = extractor.extract(messages, user_id="u1")

    assert len(result.relations) == 1
    assert result.relations[0].memory_id == result.memories[0].id


def test_extraction_empty_input(fake_llm):
    extractor = LLMExtractor(llm=fake_llm)
    result = extractor.extract([], user_id="u1")
    assert result.memories == []
    assert result.entities == []
    assert result.relations == []


def test_extraction_bad_json_returns_empty():
    class BadLLM:
        def complete(self, system, user):
            return "this is not valid json !!!"

    extractor = LLMExtractor(llm=BadLLM())
    result = extractor.extract([{"role": "user", "content": "hi"}], user_id="u1")
    assert isinstance(result, ExtractionResult)
    assert result.memories == []
    assert result.entities == []
    assert result.relations == []


def test_extraction_markdown_fences_stripped():
    class MarkdownLLM:
        def complete(self, system, user):
            data = {
                "memories": [
                    {"content": "User likes Python", "memory_type": "semantic", "importance": 0.7}
                ],
                "entities": [{"name": "User", "entity_type": "PERSON", "aliases": []}],
                "relations": [],
            }
            return f"```json\n{json.dumps(data)}\n```"

    extractor = LLMExtractor(llm=MarkdownLLM())
    result = extractor.extract([{"role": "user", "content": "I like Python"}], user_id="u1")
    assert len(result.memories) == 1
    assert result.memories[0].content == "User likes Python"


def test_extraction_window_limits_messages():
    call_log = []

    class LoggingLLM:
        def complete(self, system, user):
            call_log.append(user)
            return json.dumps({"memories": [], "entities": [], "relations": []})

    extractor = LLMExtractor(llm=LoggingLLM(), window=2)
    messages = [
        {"role": "user", "content": f"msg{i}"} for i in range(5)
    ]
    extractor.extract(messages, user_id="u1")

    assert len(call_log) == 1
    # Only last 2 messages should appear
    assert "msg3" in call_log[0]
    assert "msg4" in call_log[0]
    assert "msg0" not in call_log[0]
