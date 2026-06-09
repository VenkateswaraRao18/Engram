from __future__ import annotations

import pytest
from engram.models import Memory, Entity, Relation, SearchResult, ExtractionResult, MemoryType


def test_memory_defaults():
    m = Memory(user_id="u1", content="Test memory", memory_type=MemoryType.SEMANTIC)
    assert m.id is not None
    assert len(m.id) == 36  # UUID
    assert m.importance == 0.5
    assert m.access_count == 0
    assert m.embedding is None
    assert m.valid_until is None
    assert m.superseded_by is None
    assert m.source_message_ids == []
    assert m.metadata == {}


def test_memory_type_enum():
    assert MemoryType.EPISODIC.value == "episodic"
    assert MemoryType.SEMANTIC.value == "semantic"
    assert MemoryType.PROCEDURAL.value == "procedural"


def test_memory_with_embedding():
    emb = [0.1, 0.2, 0.3]
    m = Memory(user_id="u1", content="Test", memory_type=MemoryType.EPISODIC, embedding=emb)
    assert m.embedding == emb


def test_entity_defaults():
    e = Entity(user_id="u1", name="Alice", entity_type="PERSON")
    assert e.id is not None
    assert e.aliases == []


def test_relation_defaults():
    r = Relation(
        user_id="u1",
        source_entity_id="e1",
        relation_type="LIVES_IN",
        target_entity_id="e2",
        memory_id="m1",
    )
    assert r.id is not None
    assert r.confidence == 0.8
    assert r.valid_until is None


def test_search_result():
    m = Memory(user_id="u1", content="Test", memory_type=MemoryType.SEMANTIC)
    sr = SearchResult(memory=m, score=0.9, retrieval_path="vector")
    assert sr.score == 0.9
    assert sr.retrieval_path == "vector"
    assert sr.vector_score is None
    assert sr.graph_score is None


def test_extraction_result_defaults():
    er = ExtractionResult()
    assert er.memories == []
    assert er.entities == []
    assert er.relations == []
