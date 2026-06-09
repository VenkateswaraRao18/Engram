from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from ...models import Entity, Relation


@runtime_checkable
class GraphStore(Protocol):
    def upsert_entities(self, entities: list[Entity]) -> dict[str, str]:
        """Insert or find existing entities. Returns mapping of input_id -> canonical_id."""
        ...

    def upsert_relations(self, relations: list[Relation]) -> None:
        """Insert relations into the graph."""
        ...

    def invalidate_relation(self, relation_id: str, at: Optional[datetime] = None) -> None:
        """Mark a relation as invalid by setting valid_until."""
        ...

    def neighborhood(
        self,
        user_id: str,
        entity_names: list[str],
        hops: int = 2,
        valid_at: Optional[datetime] = None,
    ) -> list[Relation]:
        """BFS neighborhood from named entities, return valid relations."""
        ...

    def find_conflicting(self, relation: Relation) -> list[Relation]:
        """Find currently-valid relations that conflict with the given one (same functional type)."""
        ...
