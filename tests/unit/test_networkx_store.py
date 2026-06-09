from __future__ import annotations

import os
from datetime import datetime

import pytest

from engram.models import Entity, Relation
from engram.stores.graph.networkx_store import NetworkXGraphStore


@pytest.fixture
def store():
    return NetworkXGraphStore()


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
    e1 = make_entity("u1", "Alice")
    mapping = store.upsert_entities([e1])
    assert mapping[e1.id] == e1.id
    assert e1.id in store._entities


def test_upsert_entities_deduplicates_case_insensitive(store):
    e1 = make_entity("u1", "Alice")
    mapping1 = store.upsert_entities([e1])

    # Second entity with same name (different case)
    e2 = make_entity("u1", "alice")
    mapping2 = store.upsert_entities([e2])

    # e2 should map to e1's id (the canonical one)
    assert mapping2[e2.id] == e1.id
    assert len(store._entities) == 1


def test_upsert_entities_different_users_not_deduped(store):
    e1 = make_entity("u1", "Alice")
    e2 = make_entity("u2", "Alice")
    mapping = store.upsert_entities([e1, e2])
    assert mapping[e1.id] == e1.id
    assert mapping[e2.id] == e2.id
    assert len(store._entities) == 2


def test_upsert_relations(store):
    user_entity = make_entity("u1", "User")
    tampa = make_entity("u1", "Tampa", entity_type="PLACE")
    mapping = store.upsert_entities([user_entity, tampa])

    rel = make_relation("u1", mapping[user_entity.id], "LIVES_IN", mapping[tampa.id])
    store.upsert_relations([rel])

    assert rel.id in store._relations


def test_find_conflicting_lives_in(store):
    user_entity = make_entity("u1", "User")
    tampa = make_entity("u1", "Tampa", entity_type="PLACE")
    austin = make_entity("u1", "Austin", entity_type="PLACE")
    mapping = store.upsert_entities([user_entity, tampa, austin])

    user_id = mapping[user_entity.id]
    tampa_id = mapping[tampa.id]
    austin_id = mapping[austin.id]

    rel_tampa = make_relation("u1", user_id, "LIVES_IN", tampa_id, memory_id="mem1")
    store.upsert_relations([rel_tampa])

    rel_austin = make_relation("u1", user_id, "LIVES_IN", austin_id, memory_id="mem2")
    conflicts = store.find_conflicting(rel_austin)

    assert len(conflicts) == 1
    assert conflicts[0].id == rel_tampa.id


def test_find_conflicting_non_functional_type(store):
    user_entity = make_entity("u1", "User")
    alice = make_entity("u1", "Alice")
    mapping = store.upsert_entities([user_entity, alice])

    user_id = mapping[user_entity.id]
    alice_id = mapping[alice.id]

    rel1 = make_relation("u1", user_id, "KNOWS", alice_id, memory_id="mem1")
    store.upsert_relations([rel1])

    rel2 = make_relation("u1", user_id, "KNOWS", alice_id, memory_id="mem2")
    conflicts = store.find_conflicting(rel2)
    assert conflicts == []


def test_invalidate_relation(store):
    user_entity = make_entity("u1", "User")
    tampa = make_entity("u1", "Tampa", entity_type="PLACE")
    mapping = store.upsert_entities([user_entity, tampa])

    user_id = mapping[user_entity.id]
    tampa_id = mapping[tampa.id]

    rel = make_relation("u1", user_id, "LIVES_IN", tampa_id)
    store.upsert_relations([rel])

    assert store._relations[rel.id].valid_until is None

    now = datetime.utcnow()
    store.invalidate_relation(rel.id, at=now)

    assert store._relations[rel.id].valid_until is not None


def test_neighborhood_basic(store):
    user_entity = make_entity("u1", "User")
    tampa = make_entity("u1", "Tampa", entity_type="PLACE")
    mapping = store.upsert_entities([user_entity, tampa])

    user_id = mapping[user_entity.id]
    tampa_id = mapping[tampa.id]

    rel = make_relation("u1", user_id, "LIVES_IN", tampa_id)
    store.upsert_relations([rel])

    rels = store.neighborhood("u1", ["User"], hops=1)
    assert len(rels) == 1
    assert rels[0].id == rel.id


def test_neighborhood_excludes_invalidated(store):
    user_entity = make_entity("u1", "User")
    tampa = make_entity("u1", "Tampa", entity_type="PLACE")
    mapping = store.upsert_entities([user_entity, tampa])

    user_id = mapping[user_entity.id]
    tampa_id = mapping[tampa.id]

    rel = make_relation("u1", user_id, "LIVES_IN", tampa_id)
    store.upsert_relations([rel])
    store.invalidate_relation(rel.id)

    rels = store.neighborhood("u1", ["User"], hops=2)
    assert rels == []


def test_persistence(tmp_path):
    graph_path = str(tmp_path / "graph.json")

    store1 = NetworkXGraphStore(path=graph_path)
    user_entity = make_entity("u1", "User")
    tampa = make_entity("u1", "Tampa", entity_type="PLACE")
    mapping = store1.upsert_entities([user_entity, tampa])

    user_id = mapping[user_entity.id]
    tampa_id = mapping[tampa.id]

    rel = make_relation("u1", user_id, "LIVES_IN", tampa_id)
    store1.upsert_relations([rel])

    assert os.path.exists(graph_path)

    # Load fresh store from file
    store2 = NetworkXGraphStore(path=graph_path)
    assert user_id in store2._entities
    assert tampa_id in store2._entities
    assert rel.id in store2._relations
    assert store2._relations[rel.id].relation_type == "LIVES_IN"
