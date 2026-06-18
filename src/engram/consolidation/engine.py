from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from ..config import EngramConfig
from ..embeddings.base import Embedder
from ..models import ExtractionResult, Memory
from ..stores.graph.base import GraphStore
from ..stores.vector.base import VectorStore

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Handles deduplication, supersession, and storage of extracted memories."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        graph_store: GraphStore,
        config: EngramConfig,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._config = config

    def process_extraction(
        self, extraction_result: ExtractionResult, user_id: str
    ) -> list[Memory]:
        """
        Process an ExtractionResult:
        1. Embed all new memories
        2. Deduplicate against existing store
        3. Store non-duplicate memories
        4. Resolve entity IDs via graph store
        5. Handle conflicting (functional) relations → supersede old memory
        6. Insert new relations
        Returns the list of newly stored memories.
        """
        now = datetime.utcnow()

        # Step 1: Embed all new memories
        for mem in extraction_result.memories:
            mem.embedding = self._embedder.embed(mem.content)

        # Memories linked to functional (supersedable) relations must bypass dedup:
        # embedding models treat "User lives in Austin" and "User lives in Tampa" as
        # near-identical (cosine ≈ 1.0) because they share the same semantic template.
        # Without this exception, the update memory is rejected as a duplicate while
        # the old memory gets superseded — leaving no valid memory at all.
        # Functional (single-valued) relation types: a user can only have one value
        # at a time for these. New values supersede old ones rather than coexisting.
        # Also bypasses dedup because embedding models score same-template sentences
        # (e.g. "User codes in Go" vs "User codes in Rust") as near-identical.
        _FUNCTIONAL_TYPES = {
            "LIVES_IN", "WORKS_AT", "MARRIED_TO", "BORN_IN",
            "STUDIES", "USES_LANGUAGE", "DOES_EXERCISE", "RELATIONSHIP_STATUS",
        }
        functional_memory_ids = {
            rel.memory_id
            for rel in extraction_result.relations
            if rel.relation_type in _FUNCTIONAL_TYPES
        }

        # Step 2: Dedup check
        memories_to_store: list[Memory] = []
        for mem in extraction_result.memories:
            if mem.embedding is None:
                memories_to_store.append(mem)
                continue

            if mem.id in functional_memory_ids:
                # Always store updates to functional facts; supersession will
                # invalidate the stale version rather than dedup dropping the new one.
                memories_to_store.append(mem)
                continue

            similar = self._vector_store.search(
                user_id,
                mem.embedding,
                k=self._config.dedup_top_k,
                filter_valid=True,
            )
            is_duplicate = any(
                score >= self._config.dedup_similarity_threshold
                for _, score in similar
            )
            if not is_duplicate:
                memories_to_store.append(mem)
            else:
                logger.debug("Skipping duplicate memory: %s", mem.content)

        # Step 3: Store non-duplicate memories
        stored_memory_ids = set()
        if memories_to_store:
            self._vector_store.add(memories_to_store)
            stored_memory_ids = {m.id for m in memories_to_store}

        # Step 4: Resolve entity IDs
        id_mapping = self._graph_store.upsert_entities(extraction_result.entities)

        # Step 5 & 6: Handle relations
        for relation in extraction_result.relations:
            # Remap entity IDs to canonical IDs
            relation.source_entity_id = id_mapping.get(
                relation.source_entity_id, relation.source_entity_id
            )
            relation.target_entity_id = id_mapping.get(
                relation.target_entity_id, relation.target_entity_id
            )

            # For functional (single-valued) relations, supersede as long as
            # any memory from this batch was stored. The functional memory always
            # bypasses dedup, but relation.memory_id may point to a different
            # (non-functional) memory that got deduped, so a strict id check
            # would silently skip supersession.
            # For non-functional relations, keep the strict check to avoid
            # invalidating old memories when the new one was a duplicate.
            if relation.relation_type in _FUNCTIONAL_TYPES:
                should_supersede = bool(stored_memory_ids)
            else:
                should_supersede = relation.memory_id in stored_memory_ids

            if should_supersede:
                conflicts = self._graph_store.find_conflicting(relation)
                for conflict in conflicts:
                    self._graph_store.invalidate_relation(conflict.id, at=now)
                    self._vector_store.update_metadata(
                        conflict.memory_id,
                        valid_until=now,
                        superseded_by=relation.memory_id,
                    )
                    logger.debug(
                        "Superseded memory %s with %s", conflict.memory_id, relation.memory_id
                    )

            # Insert new relation regardless of whether memory was stored
            self._graph_store.upsert_relations([relation])

        return memories_to_store
