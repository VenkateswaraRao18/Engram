from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime
from typing import Optional

import networkx as nx

from ...models import Entity, Relation

FUNCTIONAL_TYPES = {
    "LIVES_IN", "WORKS_AT", "BORN_IN", "MARRIED_TO",
    "STUDIES", "USES_LANGUAGE", "DOES_EXERCISE", "RELATIONSHIP_STATUS",
}


class NetworkXGraphStore:
    """In-memory graph store backed by NetworkX with optional JSON persistence."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()

        if path and os.path.exists(path):
            self._load()

    def upsert_entities(self, entities: list[Entity]) -> dict[str, str]:
        """
        Insert or find existing entities by case-insensitive name + user_id.
        Returns dict mapping input entity id -> canonical entity id.
        """
        id_mapping: dict[str, str] = {}

        for entity in entities:
            canonical_id = self._find_entity(entity.user_id, entity.name)
            if canonical_id is not None:
                # Already exists; map input id to existing canonical id
                id_mapping[entity.id] = canonical_id
            else:
                # New entity
                self._entities[entity.id] = entity
                self._graph.add_node(entity.id, name=entity.name, entity_type=entity.entity_type)
                id_mapping[entity.id] = entity.id

        self._save()
        return id_mapping

    def _find_entity(self, user_id: str, name: str) -> Optional[str]:
        """Find existing entity by case-insensitive name and user_id. Returns id or None."""
        name_lower = name.lower()
        for eid, ent in self._entities.items():
            if ent.user_id == user_id and ent.name.lower() == name_lower:
                return eid
        return None

    def upsert_relations(self, relations: list[Relation]) -> None:
        """Store relations and add edges to the graph."""
        for rel in relations:
            self._relations[rel.id] = rel
            self._graph.add_edge(
                rel.source_entity_id,
                rel.target_entity_id,
                key=rel.id,
                relation_id=rel.id,
                relation_type=rel.relation_type,
            )
        self._save()

    def invalidate_relation(self, relation_id: str, at: Optional[datetime] = None) -> None:
        """Set valid_until on the relation."""
        if relation_id in self._relations:
            rel = self._relations[relation_id]
            invalidated = rel.model_copy(update={"valid_until": at or datetime.utcnow()})
            self._relations[relation_id] = invalidated
        self._save()

    def neighborhood(
        self,
        user_id: str,
        entity_names: list[str],
        hops: int = 2,
        valid_at: Optional[datetime] = None,
    ) -> list[Relation]:
        """BFS from named entities, collect valid relations within `hops`."""
        if valid_at is None:
            valid_at = datetime.utcnow()

        # Find starting entity node IDs
        start_ids: set[str] = set()
        for name in entity_names:
            eid = self._find_entity(user_id, name)
            if eid is not None:
                start_ids.add(eid)

        if not start_ids:
            return []

        visited_nodes: set[str] = set()
        queue: deque = deque()
        for sid in start_ids:
            queue.append((sid, 0))
            visited_nodes.add(sid)

        collected_relations: dict[str, Relation] = {}

        while queue:
            node_id, depth = queue.popleft()
            if depth >= hops:
                continue

            # Outgoing edges
            for _, tgt, key, data in self._graph.out_edges(node_id, keys=True, data=True):
                rel_id = data.get("relation_id")
                if rel_id and rel_id in self._relations:
                    rel = self._relations[rel_id]
                    if rel.user_id == user_id and self._is_valid(rel, valid_at):
                        collected_relations[rel_id] = rel
                if tgt not in visited_nodes:
                    visited_nodes.add(tgt)
                    queue.append((tgt, depth + 1))

            # Incoming edges
            for src, _, key, data in self._graph.in_edges(node_id, keys=True, data=True):
                rel_id = data.get("relation_id")
                if rel_id and rel_id in self._relations:
                    rel = self._relations[rel_id]
                    if rel.user_id == user_id and self._is_valid(rel, valid_at):
                        collected_relations[rel_id] = rel
                if src not in visited_nodes:
                    visited_nodes.add(src)
                    queue.append((src, depth + 1))

        return list(collected_relations.values())

    def find_conflicting(self, relation: Relation) -> list[Relation]:
        """Find currently-valid relations that conflict with functional relation types."""
        if relation.relation_type not in FUNCTIONAL_TYPES:
            return []

        now = datetime.utcnow()
        conflicts = []
        for rel in self._relations.values():
            if (
                rel.id != relation.id
                and rel.user_id == relation.user_id
                and rel.source_entity_id == relation.source_entity_id
                and rel.relation_type == relation.relation_type
                and self._is_valid(rel, now)
            ):
                conflicts.append(rel)
        return conflicts

    def _is_valid(self, rel: Relation, at: datetime) -> bool:
        """Check if a relation is valid at the given time."""
        return rel.valid_until is None or rel.valid_until > at

    def _save(self) -> None:
        """Atomically write state to JSON file if path is set."""
        if not self.path:
            return

        data = {
            "entities": {
                eid: {
                    **ent.model_dump(),
                }
                for eid, ent in self._entities.items()
            },
            "relations": {},
        }

        # Serialize relations with datetime as isoformat
        for rid, rel in self._relations.items():
            rel_dict = rel.model_dump()
            # Convert datetime fields to strings
            for field in ("valid_from", "valid_until"):
                val = rel_dict.get(field)
                if isinstance(val, datetime):
                    rel_dict[field] = val.isoformat()
                # Already string or None is fine
            data["relations"][rid] = rel_dict

        # Also handle entity datetime fields if any (none currently, but be safe)
        for eid in data["entities"]:
            ent_dict = data["entities"][eid]
            # No datetime fields in Entity currently

        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str)
        os.replace(tmp_path, self.path)

    def _load(self) -> None:
        """Load state from JSON file."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._entities = {}
        for eid, ent_dict in data.get("entities", {}).items():
            ent = Entity(**ent_dict)
            self._entities[eid] = ent
            self._graph.add_node(eid, name=ent.name, entity_type=ent.entity_type)

        self._relations = {}
        for rid, rel_dict in data.get("relations", {}).items():
            # Parse datetime strings
            for field in ("valid_from", "valid_until"):
                val = rel_dict.get(field)
                if val is not None and isinstance(val, str):
                    rel_dict[field] = datetime.fromisoformat(val)
            rel = Relation(**rel_dict)
            self._relations[rid] = rel
            self._graph.add_edge(
                rel.source_entity_id,
                rel.target_entity_id,
                key=rel.id,
                relation_id=rel.id,
                relation_type=rel.relation_type,
            )
