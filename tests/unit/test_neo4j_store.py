from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from engram.models import Entity, Relation
from engram.stores.graph.neo4j_store import Neo4jGraphStore


@pytest.fixture(scope="module")
def store():
    """Single Neo4j store for the whole module. Skip if Neo4j unavailable."""
    try:
        s = Neo4jGraphStore()
        s._driver.verify_connectivity()
        yield s
        s.close()
    except Exception:
        pytest.skip("Neo4j not available")


def uid() -> str:
    """Unique user_id per test run to avoid state conflicts between tests."""
    return f"test_{uuid.uuid4().hex[:8]}"


def make_entity(user_id: str, name: str, entity_type: str = "PERSON") -> Entity:
    return Entity(user_id=user_id, name=name, entity_type=entity_type)


def make_relation(
    user_id: str,
    src_id: str,
    rel_type: str,
    tgt_id: str,
    memory_id: str = "mem1",
) -> Relation:
    return Relation(
        user_id=user_id,
        source_entity_id=src_id,
        relation_type=rel_type,
        target_entity_id=tgt_id,
        memory_id=memory_id,
    )


def test_upsert_entities_basic(store):
    user_id = uid()
    e1 = make_entity(user_id, "Alice")
    mapping = store.upsert_entities([e1])
    assert mapping[e1.id] == e1.id


def test_upsert_entities_deduplicates_case_insensitive(store):
    user_id = uid()
    e1 = make_entity(user_id, "Alice")
    mapping1 = store.upsert_entities([e1])

    e2 = make_entity(user_id, "alice")
    mapping2 = store.upsert_entities([e2])

    # e2 should resolve to e1's canonical id
    assert mapping2[e2.id] == mapping1[e1.id]


def test_upsert_entities_different_users_not_deduped(store):
    e1 = make_entity(uid(), "Alice")
    e2 = make_entity(uid(), "Alice")
    mapping = store.upsert_entities([e1, e2])
    assert mapping[e1.id] != mapping[e2.id]


def test_upsert_relations(store):
    user_id = uid()
    user_e = make_entity(user_id, "User")
    tampa_e = make_entity(user_id, "Tampa", "PLACE")
    mapping = store.upsert_entities([user_e, tampa_e])

    rel = make_relation(user_id, mapping[user_e.id], "LIVES_IN", mapping[tampa_e.id])
    store.upsert_relations([rel])

    # Verify via find_conflicting (finds the stored relation)
    rel2 = make_relation(user_id, mapping[user_e.id], "LIVES_IN", mapping[tampa_e.id], memory_id="mem2")
    conflicts = store.find_conflicting(rel2)
    assert any(c.id == rel.id for c in conflicts)


def test_find_conflicting_lives_in(store):
    user_id = uid()
    user_e = make_entity(user_id, "User")
    tampa_e = make_entity(user_id, "Tampa", "PLACE")
    austin_e = make_entity(user_id, "Austin", "PLACE")
    mapping = store.upsert_entities([user_e, tampa_e, austin_e])

    user_id_canonical = mapping[user_e.id]
    tampa_id = mapping[tampa_e.id]
    austin_id = mapping[austin_e.id]

    rel_tampa = make_relation(user_id, user_id_canonical, "LIVES_IN", tampa_id, "mem1")
    store.upsert_relations([rel_tampa])

    rel_austin = make_relation(user_id, user_id_canonical, "LIVES_IN", austin_id, "mem2")
    conflicts = store.find_conflicting(rel_austin)

    assert len(conflicts) == 1
    assert conflicts[0].id == rel_tampa.id


def test_find_conflicting_non_functional_type(store):
    user_id = uid()
    user_e = make_entity(user_id, "User")
    alice_e = make_entity(user_id, "Alice")
    mapping = store.upsert_entities([user_e, alice_e])

    rel1 = make_relation(user_id, mapping[user_e.id], "KNOWS", mapping[alice_e.id], "mem1")
    store.upsert_relations([rel1])

    rel2 = make_relation(user_id, mapping[user_e.id], "KNOWS", mapping[alice_e.id], "mem2")
    assert store.find_conflicting(rel2) == []


def test_invalidate_relation(store):
    user_id = uid()
    user_e = make_entity(user_id, "User")
    tampa_e = make_entity(user_id, "Tampa", "PLACE")
    mapping = store.upsert_entities([user_e, tampa_e])

    rel = make_relation(user_id, mapping[user_e.id], "LIVES_IN", mapping[tampa_e.id])
    store.upsert_relations([rel])

    now = datetime.utcnow()
    store.invalidate_relation(rel.id, at=now)

    # After invalidation, should not appear as conflicting
    rel2 = make_relation(user_id, mapping[user_e.id], "LIVES_IN", mapping[tampa_e.id], "mem2")
    conflicts = store.find_conflicting(rel2)
    assert not any(c.id == rel.id for c in conflicts)


def test_neighborhood_basic(store):
    user_id = uid()
    user_e = make_entity(user_id, "User")
    tampa_e = make_entity(user_id, "Tampa", "PLACE")
    mapping = store.upsert_entities([user_e, tampa_e])

    rel = make_relation(user_id, mapping[user_e.id], "LIVES_IN", mapping[tampa_e.id])
    store.upsert_relations([rel])

    rels = store.neighborhood(user_id, ["User"], hops=1)
    assert any(r.id == rel.id for r in rels)


def test_neighborhood_excludes_invalidated(store):
    user_id = uid()
    user_e = make_entity(user_id, "User")
    tampa_e = make_entity(user_id, "Tampa", "PLACE")
    mapping = store.upsert_entities([user_e, tampa_e])

    rel = make_relation(user_id, mapping[user_e.id], "LIVES_IN", mapping[tampa_e.id])
    store.upsert_relations([rel])
    store.invalidate_relation(rel.id)

    rels = store.neighborhood(user_id, ["User"], hops=2)
    assert not any(r.id == rel.id for r in rels)


def test_supersession_tampa_to_austin(store):
    """Full supersession scenario: Tampa LIVES_IN gets invalidated when Austin is added."""
    user_id = uid()
    user_e = make_entity(user_id, "User")
    tampa_e = make_entity(user_id, "Tampa", "PLACE")
    austin_e = make_entity(user_id, "Austin", "PLACE")
    mapping = store.upsert_entities([user_e, tampa_e, austin_e])

    uid_canon = mapping[user_e.id]
    tampa_id = mapping[tampa_e.id]
    austin_id = mapping[austin_e.id]

    # Step 1: store Tampa
    rel_tampa = make_relation(user_id, uid_canon, "LIVES_IN", tampa_id, "mem_tampa")
    store.upsert_relations([rel_tampa])

    # Step 2: detect conflict, invalidate Tampa, store Austin
    rel_austin = make_relation(user_id, uid_canon, "LIVES_IN", austin_id, "mem_austin")
    conflicts = store.find_conflicting(rel_austin)
    assert len(conflicts) == 1

    store.invalidate_relation(conflicts[0].id)
    store.upsert_relations([rel_austin])

    # Tampa should be gone from neighborhood, Austin should be present
    rels = store.neighborhood(user_id, ["User"], hops=1)
    rel_ids = [r.id for r in rels]
    assert rel_tampa.id not in rel_ids
    assert rel_austin.id in rel_ids
