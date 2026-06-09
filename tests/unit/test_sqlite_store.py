from __future__ import annotations

from datetime import datetime

import pytest

from engram.models import Memory, MemoryType
from engram.stores.vector.sqlite_store import SQLiteVectorStore


@pytest.fixture
def store():
    return SQLiteVectorStore(db_path=":memory:")


@pytest.fixture
def memory_with_embedding(fake_embedder):
    emb = fake_embedder.embed("User lives in Tampa")
    return Memory(
        user_id="u1",
        content="User lives in Tampa",
        memory_type=MemoryType.SEMANTIC,
        importance=0.85,
        embedding=emb,
    )


def test_add_and_get(store, memory_with_embedding):
    store.add([memory_with_embedding])
    retrieved = store.get(memory_with_embedding.id)
    assert retrieved is not None
    assert retrieved.id == memory_with_embedding.id
    assert retrieved.content == "User lives in Tampa"
    assert retrieved.user_id == "u1"


def test_get_nonexistent(store):
    result = store.get("nonexistent-id")
    assert result is None


def test_search_returns_similar(store, fake_embedder):
    # Add a few memories with different embeddings
    m1 = Memory(
        user_id="u1",
        content="User lives in Tampa",
        memory_type=MemoryType.SEMANTIC,
        embedding=fake_embedder.embed("User lives in Tampa"),
    )
    m2 = Memory(
        user_id="u1",
        content="User likes pizza",
        memory_type=MemoryType.SEMANTIC,
        embedding=fake_embedder.embed("User likes pizza"),
    )
    store.add([m1, m2])

    query_emb = fake_embedder.embed("User lives in Tampa")
    results = store.search("u1", query_emb, k=5)

    assert len(results) == 2
    assert results[0][0] == m1.id  # Tampa should rank higher for Tampa query
    assert results[0][1] > results[1][1]  # Scores should be sorted descending


def test_filter_valid_excludes_superseded(store, fake_embedder):
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

    query_emb = fake_embedder.embed("location")
    results_valid = store.search("u1", query_emb, k=10, filter_valid=True)
    results_all = store.search("u1", query_emb, k=10, filter_valid=False)

    assert len(results_valid) == 1
    assert results_valid[0][0] == m2.id
    assert len(results_all) == 2


def test_filter_valid_excludes_expired(store, fake_embedder):
    m1 = Memory(
        user_id="u1",
        content="Old memory",
        memory_type=MemoryType.SEMANTIC,
        embedding=fake_embedder.embed("Old memory"),
        valid_until=datetime(2020, 1, 1),
    )
    store.add([m1])

    query_emb = fake_embedder.embed("memory")
    results = store.search("u1", query_emb, k=10, filter_valid=True)
    assert len(results) == 0


def test_delete(store, memory_with_embedding):
    store.add([memory_with_embedding])
    assert store.get(memory_with_embedding.id) is not None

    store.delete([memory_with_embedding.id])
    assert store.get(memory_with_embedding.id) is None


def test_update_metadata_superseded_by(store, memory_with_embedding):
    store.add([memory_with_embedding])
    store.update_metadata(memory_with_embedding.id, superseded_by="new-mem-id")

    updated = store.get(memory_with_embedding.id)
    assert updated.superseded_by == "new-mem-id"


def test_update_metadata_datetime(store, memory_with_embedding):
    store.add([memory_with_embedding])
    now = datetime.utcnow()
    store.update_metadata(memory_with_embedding.id, valid_until=now)

    updated = store.get(memory_with_embedding.id)
    assert updated.valid_until is not None


def test_count(store, fake_embedder):
    assert store.count("u1") == 0
    m1 = Memory(
        user_id="u1",
        content="Memory 1",
        memory_type=MemoryType.SEMANTIC,
        embedding=fake_embedder.embed("Memory 1"),
    )
    m2 = Memory(
        user_id="u1",
        content="Memory 2",
        memory_type=MemoryType.SEMANTIC,
        embedding=fake_embedder.embed("Memory 2"),
    )
    store.add([m1, m2])
    assert store.count("u1") == 2


def test_get_all_valid(store, fake_embedder):
    m1 = Memory(
        user_id="u1",
        content="Valid memory",
        memory_type=MemoryType.SEMANTIC,
        embedding=fake_embedder.embed("Valid memory"),
    )
    m2 = Memory(
        user_id="u1",
        content="Superseded memory",
        memory_type=MemoryType.SEMANTIC,
        embedding=fake_embedder.embed("Superseded memory"),
        superseded_by="other",
    )
    store.add([m1, m2])
    valid = store.get_all_valid("u1")
    assert len(valid) == 1
    assert valid[0].id == m1.id
