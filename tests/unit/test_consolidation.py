from __future__ import annotations

import json

import pytest

from engram.consolidation.engine import ConsolidationEngine
from engram.extraction.llm_extractor import LLMExtractor
from engram.models import ExtractionResult, Memory, MemoryType
from engram.stores.graph.networkx_store import NetworkXGraphStore
from engram.stores.vector.sqlite_store import SQLiteVectorStore


# FakeLLM that alternates between Tampa and Austin responses
class SequentialLLM:
    def __init__(self, responses: list[dict]):
        self._responses = responses
        self._call_count = 0

    def complete(self, system: str, user: str) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return json.dumps(self._responses[idx])


TAMPA_RESPONSE = {
    "memories": [
        {"content": "User lives in Tampa", "memory_type": "semantic", "importance": 0.85}
    ],
    "entities": [
        {"name": "User", "entity_type": "PERSON", "aliases": []},
        {"name": "Tampa", "entity_type": "PLACE", "aliases": []},
    ],
    "relations": [
        {
            "source_entity_name": "User",
            "relation_type": "LIVES_IN",
            "target_entity_name": "Tampa",
            "confidence": 0.9,
        }
    ],
}

AUSTIN_RESPONSE = {
    "memories": [
        {"content": "User moved to Austin, Texas", "memory_type": "episodic", "importance": 0.85}
    ],
    "entities": [
        {"name": "User", "entity_type": "PERSON", "aliases": []},
        {"name": "Austin", "entity_type": "PLACE", "aliases": []},
    ],
    "relations": [
        {
            "source_entity_name": "User",
            "relation_type": "LIVES_IN",
            "target_entity_name": "Austin",
            "confidence": 0.95,
        }
    ],
}


@pytest.fixture
def setup_components(fake_embedder, config):
    vector_store = SQLiteVectorStore(db_path=":memory:")
    graph_store = NetworkXGraphStore()
    engine = ConsolidationEngine(
        embedder=fake_embedder,
        vector_store=vector_store,
        graph_store=graph_store,
        config=config,
    )
    return vector_store, graph_store, engine


def test_tampa_stored(fake_embedder, config, setup_components):
    vector_store, graph_store, engine = setup_components
    extractor = LLMExtractor(llm=SequentialLLM([TAMPA_RESPONSE]))

    messages = [{"role": "user", "content": "Hi, I'm Alex. I live in Tampa, Florida."}]
    extraction = extractor.extract(messages, user_id="u1")
    stored = engine.process_extraction(extraction, user_id="u1")

    assert len(stored) == 1
    assert stored[0].content == "User lives in Tampa"

    # Verify LIVES_IN(User, Tampa) in graph
    rels = graph_store.neighborhood("u1", ["User"], hops=2)
    assert any(r.relation_type == "LIVES_IN" for r in rels)


def test_tampa_then_austin_supersession(fake_embedder, config, setup_components):
    vector_store, graph_store, engine = setup_components
    seq_llm = SequentialLLM([TAMPA_RESPONSE, AUSTIN_RESPONSE])
    extractor = LLMExtractor(llm=seq_llm)

    # Step 1: Tampa
    msgs1 = [{"role": "user", "content": "Hi, I'm Alex. I live in Tampa, Florida."}]
    extraction1 = extractor.extract(msgs1, user_id="u1")
    stored1 = engine.process_extraction(extraction1, user_id="u1")
    tampa_memory_id = stored1[0].id

    # Step 2: Austin
    msgs2 = [{"role": "user", "content": "I just moved to Austin, Texas last month."}]
    extraction2 = extractor.extract(msgs2, user_id="u1")
    stored2 = engine.process_extraction(extraction2, user_id="u1")

    assert len(stored2) == 1
    assert "Austin" in stored2[0].content

    # Tampa memory should be superseded
    tampa_mem = vector_store.get(tampa_memory_id)
    assert tampa_mem is not None
    assert tampa_mem.superseded_by is not None

    # Only Austin should be valid
    valid_memories = vector_store.get_all_valid("u1")
    valid_contents = [m.content for m in valid_memories]
    assert any("Austin" in c for c in valid_contents)
    assert not any("Tampa" in c for c in valid_contents)

    # LIVES_IN(User, Austin) should be in graph; LIVES_IN(User, Tampa) invalidated
    rels = graph_store.neighborhood("u1", ["User"], hops=2)
    live_in_rels = [r for r in rels if r.relation_type == "LIVES_IN"]
    # Should have exactly 1 valid LIVES_IN (Austin)
    assert len(live_in_rels) == 1
    # Find Austin entity to verify
    austin_id = None
    for eid, ent in graph_store._entities.items():
        if ent.name.lower() == "austin":
            austin_id = eid
    assert austin_id is not None
    assert live_in_rels[0].target_entity_id == austin_id


def test_dedup_same_memory_twice(fake_embedder, config, setup_components):
    vector_store, graph_store, engine = setup_components

    # Manually create two memories with identical embeddings (to trigger dedup)
    emb = fake_embedder.embed("User lives in Tampa")
    m1 = Memory(
        user_id="u1",
        content="User lives in Tampa",
        memory_type=MemoryType.SEMANTIC,
        importance=0.85,
        embedding=emb,
    )
    # Store the first one directly
    vector_store.add([m1])

    # Now process extraction with same content
    extraction = ExtractionResult(
        memories=[
            Memory(
                user_id="u1",
                content="User lives in Tampa",
                memory_type=MemoryType.SEMANTIC,
                importance=0.85,
                embedding=emb,
            )
        ],
        entities=[],
        relations=[],
    )
    # Set high threshold to ensure dedup detection
    config.dedup_similarity_threshold = 0.99
    engine2 = ConsolidationEngine(
        embedder=fake_embedder,
        vector_store=vector_store,
        graph_store=graph_store,
        config=config,
    )
    stored = engine2.process_extraction(extraction, user_id="u1")

    # Should not store the duplicate
    assert len(stored) == 0
    assert vector_store.count("u1") == 1


def test_empty_extraction_no_crash(fake_embedder, config, setup_components):
    _, _, engine = setup_components
    extraction = ExtractionResult()
    stored = engine.process_extraction(extraction, user_id="u1")
    assert stored == []
