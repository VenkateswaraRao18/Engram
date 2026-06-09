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

        # Step 2: Dedup check
        memories_to_store: list[Memory] = []
        for mem in extraction_result.memories:
            if mem.embedding is None:
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
        if memories_to_store:
            self._vector_store.add(memories_to_store)

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

            # Find and resolve conflicts
            conflicts = self._graph_store.find_conflicting(relation)
            for conflict in conflicts:
                # Invalidate old relation
                self._graph_store.invalidate_relation(conflict.id, at=now)
                # Supersede the old memory
                self._vector_store.update_metadata(
                    conflict.memory_id,
                    valid_until=now,
                    superseded_by=relation.memory_id,
                )
                logger.debug(
                    "Superseded memory %s with %s", conflict.memory_id, relation.memory_id
                )

            # Insert new relation
            self._graph_store.upsert_relations([relation])

        return memories_to_store
