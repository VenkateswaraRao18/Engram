from __future__ import annotations

from datetime import datetime

import pytest

from engram.models import Memory, MemoryType
from engram.stores.vector.faiss_store import FAISSVectorStore


@pytest.fixture
def store():
    return FAISSVectorStore(db_path=":memory:", dimensions=768)


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


def test_get_nonexistent(store):
    assert store.get("nonexistent-id") is None


def test_search_returns_similar(store, fake_embedder):
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

    results = store.search("u1", fake_embedder.embed("User lives in Tampa"), k=5)

    assert len(results) == 2
    assert results[0][0] == m1.id
    assert results[0][1] > results[1][1]


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

    query = fake_embedder.embed("location")
    valid = store.search("u1", query, k=10, filter_valid=True)
    all_ = store.search("u1", query, k=10, filter_valid=False)

    assert len(valid) == 1
    assert valid[0][0] == m2.id
    assert len(all_) == 2


def test_filter_valid_excludes_expired(store, fake_embedder):
    m = Memory(
        user_id="u1",
        content="Old memory",
        memory_type=MemoryType.SEMANTIC,
        embedding=fake_embedder.embed("Old memory"),
        valid_until=datetime(2020, 1, 1),
    )
    store.add([m])
    results = store.search("u1", fake_embedder.embed("memory"), k=10, filter_valid=True)
    assert len(results) == 0


def test_delete(store, memory_with_embedding):
    store.add([memory_with_embedding])
    store.delete([memory_with_embedding.id])
    assert store.get(memory_with_embedding.id) is None


def test_delete_removes_from_faiss(store, fake_embedder):
    m = Memory(
        user_id="u1",
        content="User lives in Tampa",
        memory_type=MemoryType.SEMANTIC,
        embedding=fake_embedder.embed("User lives in Tampa"),
    )
    store.add([m])
    assert store._index.ntotal == 1

    store.delete([m.id])
    assert store._index.ntotal == 0


def test_update_metadata_superseded_by(store, memory_with_embedding):
    store.add([memory_with_embedding])
    store.update_metadata(memory_with_embedding.id, superseded_by="new-mem-id")
    updated = store.get(memory_with_embedding.id)
    assert updated.superseded_by == "new-mem-id"


def test_update_metadata_datetime(store, memory_with_embedding):
    store.add([memory_with_embedding])
    store.update_metadata(memory_with_embedding.id, valid_until=datetime.utcnow())
    updated = store.get(memory_with_embedding.id)
    assert updated.valid_until is not None


def test_count(store, fake_embedder):
    assert store.count("u1") == 0
    store.add([
        Memory(user_id="u1", content="M1", memory_type=MemoryType.SEMANTIC,
               embedding=fake_embedder.embed("M1")),
        Memory(user_id="u1", content="M2", memory_type=MemoryType.SEMANTIC,
               embedding=fake_embedder.embed("M2")),
    ])
    assert store.count("u1") == 2


def test_get_all_valid(store, fake_embedder):
    m1 = Memory(user_id="u1", content="Valid", memory_type=MemoryType.SEMANTIC,
                embedding=fake_embedder.embed("Valid"))
    m2 = Memory(user_id="u1", content="Superseded", memory_type=MemoryType.SEMANTIC,
                embedding=fake_embedder.embed("Superseded"), superseded_by="other")
    store.add([m1, m2])
    valid = store.get_all_valid("u1")
    assert len(valid) == 1
    assert valid[0].id == m1.id


def test_faiss_index_persists_after_add(store, fake_embedder):
    for i in range(5):
        store.add([Memory(
            user_id="u1", content=f"Memory {i}",
            memory_type=MemoryType.SEMANTIC,
            embedding=fake_embedder.embed(f"Memory {i}"),
        )])
    assert store._index.ntotal == 5
