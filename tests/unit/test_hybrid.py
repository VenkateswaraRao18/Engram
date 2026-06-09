from __future__ import annotations

import pytest

from engram.models import Memory, MemoryType
from engram.retrieval.hybrid import HybridRetriever, _reciprocal_rank_fusion
from engram.stores.graph.networkx_store import NetworkXGraphStore
from engram.stores.vector.sqlite_store import SQLiteVectorStore


def test_rrf_single_list():
    """RRF with only vector results."""
    vector = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    scores = _reciprocal_rank_fusion(vector, [], k=60)
    # a: 1/(60+1) = 1/61; b: 1/62; c: 1/63
    assert abs(scores["a"] - 1 / 61) < 1e-9
    assert abs(scores["b"] - 1 / 62) < 1e-9
    assert abs(scores["c"] - 1 / 63) < 1e-9
    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_both_lists():
    """RRF with overlapping results gets higher score."""
    vector = [("a", 0.9), ("b", 0.8)]
    graph = [("b", 0.9), ("c", 0.7)]
    scores = _reciprocal_rank_fusion(vector, graph, k=60)
    # "b" appears in both lists so it should have higher score than "a" (only in vector)
    assert scores["b"] > scores["a"]
    assert "c" in scores


def test_rrf_no_results():
    scores = _reciprocal_rank_fusion([], [], k=60)
    assert scores == {}


def test_search_returns_sorted_results(fake_embedder, config):
    store = SQLiteVectorStore(db_path=":memory:")
    graph = NetworkXGraphStore()

    m1 = Memory(
        user_id="u1",
        content="User lives in Tampa Florida",
        memory_type=MemoryType.SEMANTIC,
        importance=0.85,
        embedding=fake_embedder.embed("User lives in Tampa Florida"),
    )
    m2 = Memory(
        user_id="u1",
        content="User likes to cook pasta",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        embedding=fake_embedder.embed("User likes to cook pasta"),
    )
    store.add([m1, m2])

    retriever = HybridRetriever(
        embedder=fake_embedder,
        vector_store=store,
        graph_store=graph,
        config=config,
    )

    results = retriever.search("Tampa Florida location", "u1", k=5)
    assert len(results) > 0
    # Scores should be sorted descending
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_search_filter_valid_excludes_superseded(fake_embedder, config):
    store = SQLiteVectorStore(db_path=":memory:")
    graph = NetworkXGraphStore()

    m1 = Memory(
        user_id="u1",
        content="User lives in Tampa",
        memory_type=MemoryType.SEMANTIC,
        embedding=fake_embedder.embed("User lives in Tampa"),
        superseded_by="some-other-id",
    )
    m2 = Memory(
        user_id="u1",
        content="User lives in Austin",
        memory_type=MemoryType.SEMANTIC,
        embedding=fake_embedder.embed("User lives in Austin"),
    )
    store.add([m1, m2])

    retriever = HybridRetriever(
        embedder=fake_embedder,
        vector_store=store,
        graph_store=graph,
        config=config,
    )

    results = retriever.search("Where does user live", "u1", k=5)
    ids = [r.memory.id for r in results]
    assert m1.id not in ids
    assert m2.id in ids


def test_search_empty_store(fake_embedder, config):
    store = SQLiteVectorStore(db_path=":memory:")
    graph = NetworkXGraphStore()

    retriever = HybridRetriever(
        embedder=fake_embedder,
        vector_store=store,
        graph_store=graph,
        config=config,
    )

    results = retriever.search("anything", "u1", k=5)
    assert results == []
