from __future__ import annotations

from datetime import datetime
from typing import Optional

from neo4j import GraphDatabase

from ...models import Entity, Relation

FUNCTIONAL_TYPES = {
    "LIVES_IN", "WORKS_AT", "BORN_IN", "MARRIED_TO",
    "STUDIES", "USES_LANGUAGE", "DOES_EXERCISE", "RELATIONSHIP_STATUS",
}


class Neo4jGraphStore:
    """Graph store backed by Neo4j with Cypher queries and temporal validity."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: Optional[str] = None,
    ):
        auth = (user, password) if password else None
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._setup_constraints()

    def _setup_constraints(self) -> None:
        with self._driver.session() as session:
            try:
                session.run(
                    "CREATE CONSTRAINT engram_entity_unique IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE (e.user_id, e.name_lower) IS UNIQUE"
                )
            except Exception:
                pass  # constraint may already exist

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def upsert_entities(self, entities: list[Entity]) -> dict[str, str]:
        """MERGE entities on (user_id, name_lower). Returns input_id → canonical_id mapping."""
        id_mapping: dict[str, str] = {}
        with self._driver.session() as session:
            for entity in entities:
                result = session.run(
                    """
                    MERGE (e:Entity {user_id: $user_id, name_lower: $name_lower})
                    ON CREATE SET
                        e.id          = $id,
                        e.name        = $name,
                        e.entity_type = $entity_type,
                        e.aliases     = $aliases
                    RETURN e.id AS canonical_id
                    """,
                    user_id=entity.user_id,
                    name_lower=entity.name.lower(),
                    id=entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    aliases=entity.aliases,
                )
                record = result.single()
                id_mapping[entity.id] = record["canonical_id"]
        return id_mapping

    def upsert_relations(self, relations: list[Relation]) -> None:
        """Create RELATES_TO edges. Source/target entity IDs must already exist."""
        with self._driver.session() as session:
            for rel in relations:
                session.run(
                    """
                    MATCH (src:Entity {id: $source_entity_id})
                    MATCH (tgt:Entity {id: $target_entity_id})
                    CREATE (src)-[r:RELATES_TO {
                        id:               $id,
                        user_id:          $user_id,
                        relation_type:    $relation_type,
                        source_entity_id: $source_entity_id,
                        target_entity_id: $target_entity_id,
                        memory_id:        $memory_id,
                        valid_from:       $valid_from,
                        valid_until:      null,
                        confidence:       $confidence
                    }]->(tgt)
                    """,
                    id=rel.id,
                    user_id=rel.user_id,
                    relation_type=rel.relation_type,
                    source_entity_id=rel.source_entity_id,
                    target_entity_id=rel.target_entity_id,
                    memory_id=rel.memory_id,
                    valid_from=rel.valid_from.isoformat(),
                    confidence=rel.confidence,
                )

    def invalidate_relation(self, relation_id: str, at: Optional[datetime] = None) -> None:
        """Set valid_until on a relation to mark it as no longer active."""
        at = at or datetime.utcnow()
        with self._driver.session() as session:
            session.run(
                """
                MATCH ()-[r:RELATES_TO {id: $relation_id}]->()
                SET r.valid_until = $valid_until
                """,
                relation_id=relation_id,
                valid_until=at.isoformat(),
            )

    def neighborhood(
        self,
        user_id: str,
        entity_names: list[str],
        hops: int = 2,
        valid_at: Optional[datetime] = None,
    ) -> list[Relation]:
        """Traverse up to `hops` from named entities, return currently-valid relations."""
        if not entity_names:
            return []

        valid_at = valid_at or datetime.utcnow()
        name_lowers = [n.lower() for n in entity_names]

        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (start:Entity)
                WHERE start.user_id = $user_id AND toLower(start.name) IN $name_lowers
                MATCH path = (start)-[:RELATES_TO*1..2]-(n)
                UNWIND relationships(path) AS r
                WITH DISTINCT r
                WHERE r.user_id = $user_id
                  AND (r.valid_until IS NULL OR r.valid_until > $valid_at)
                RETURN properties(r) AS props
                """,
                user_id=user_id,
                name_lowers=name_lowers,
                valid_at=valid_at.isoformat(),
            )
            return [self._props_to_relation(record["props"]) for record in result]

    def find_conflicting(self, relation: Relation) -> list[Relation]:
        """Return active relations with same (user_id, source_entity_id, relation_type)."""
        if relation.relation_type not in FUNCTIONAL_TYPES:
            return []

        now = datetime.utcnow().isoformat()
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (src:Entity {id: $source_entity_id})-[r:RELATES_TO]->()
                WHERE r.user_id      = $user_id
                  AND r.relation_type = $relation_type
                  AND r.id           <> $relation_id
                  AND (r.valid_until IS NULL OR r.valid_until > $now)
                RETURN properties(r) AS props
                """,
                source_entity_id=relation.source_entity_id,
                user_id=relation.user_id,
                relation_type=relation.relation_type,
                relation_id=relation.id,
                now=now,
            )
            return [self._props_to_relation(record["props"]) for record in result]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _props_to_relation(self, props: dict) -> Relation:
        return Relation(
            id=props["id"],
            user_id=props["user_id"],
            source_entity_id=props["source_entity_id"],
            relation_type=props["relation_type"],
            target_entity_id=props["target_entity_id"],
            memory_id=props["memory_id"],
            valid_from=datetime.fromisoformat(props["valid_from"]),
            valid_until=datetime.fromisoformat(props["valid_until"]) if props.get("valid_until") else None,
            confidence=float(props.get("confidence", 0.8)),
        )
