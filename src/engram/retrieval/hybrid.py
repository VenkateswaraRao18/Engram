from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from ..config import EngramConfig
from ..embeddings.base import Embedder
from ..models import Memory, SearchResult
from ..stores.graph.base import GraphStore
from ..stores.vector.base import VectorStore

# Words to ignore when extracting entity names from queries
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "through", "during",
    "where", "when", "how", "what", "who", "whom", "which", "whose",
    "that", "this", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its",
    "they", "them", "their", "live", "lives", "does", "alex",
}


def _reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    graph_results: list[tuple[str, float]],
    k: int = 60,
) -> dict[str, float]:
    """Combine vector and graph results using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for rank, (mid, _) in enumerate(vector_results):
        scores[mid] = scores.get(mid, 0) + 1.0 / (k + rank + 1)
    for rank, (mid, _) in enumerate(graph_results):
        scores[mid] = scores.get(mid, 0) + 1.0 / (k + rank + 1)
    return scores


def _extract_entity_names(query: str) -> list[str]:
    """Simple heuristic: words with initial capital, length > 2, not stop words."""
    words = query.split()
    names = []
    for word in words:
        # Strip punctuation
        clean = word.strip(".,!?;:'\"()")
        if (
            len(clean) > 2
            and clean[0].isupper()
            and clean.lower() not in _STOP_WORDS
        ):
            names.append(clean)
    return names


def _recency_decay(created_at: datetime, half_life_days: float) -> float:
    """Exponential decay based on age: exp(-ln(2) * days / half_life)."""
    now = datetime.utcnow()
    delta = now - created_at
    days = delta.total_seconds() / 86400.0
    return math.exp(-math.log(2) * days / half_life_days)


class HybridRetriever:
    """Hybrid retrieval combining vector search + graph neighborhood via RRF."""

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

    def search(self, query: str, user_id: str, k: int = 5) -> list[SearchResult]:
        """Hybrid search: vector + graph neighborhood, fused via RRF with recency/importance boost."""
        # 1. Embed query
        query_embedding = self._embedder.embed(query)

        # 2. Vector candidates
        vector_candidates = self._vector_store.search(
            user_id, query_embedding, k=20, filter_valid=True
        )
        vector_ids = {mid for mid, _ in vector_candidates}

        # 3. Extract entity names and get graph neighborhood
        entity_names = _extract_entity_names(query)
        graph_rels = self._graph_store.neighborhood(user_id, entity_names, hops=2)

        # 4. Build graph candidates from relation memory_ids
        graph_candidates: list[tuple[str, float]] = []
        seen_graph_ids: set[str] = set()
        for rel in graph_rels:
            if rel.memory_id and rel.memory_id not in seen_graph_ids:
                graph_candidates.append((rel.memory_id, rel.confidence))
                seen_graph_ids.add(rel.memory_id)

        # 5. RRF fusion
        fused = _reciprocal_rank_fusion(
            vector_candidates, graph_candidates, k=self._config.rrf_k
        )

        if not fused:
            return []

        # 6. Load all valid memories to apply boosts
        all_valid = self._vector_store.get_all_valid(user_id)
        memory_map: dict[str, Memory] = {m.id: m for m in all_valid}

        # 7. Apply recency + importance boost
        final_scores: dict[str, float] = {}
        for mid, rrf_score in fused.items():
            mem = memory_map.get(mid)
            if mem is None:
                continue
            decay = _recency_decay(mem.created_at, self._config.recency_half_life_days)
            boost = (1 + self._config.recency_weight * decay) * (
                1 + self._config.importance_weight * mem.importance
            )
            final_scores[mid] = rrf_score * boost

        # 8. Sort by final score, take top k
        sorted_ids = sorted(final_scores, key=lambda x: final_scores[x], reverse=True)[:k]

        # 9. Build SearchResult objects
        results: list[SearchResult] = []
        graph_ids = seen_graph_ids

        for mid in sorted_ids:
            mem = memory_map.get(mid)
            if mem is None:
                continue

            in_vector = mid in vector_ids
            in_graph = mid in graph_ids

            if in_vector and in_graph:
                path = "both"
            elif in_graph:
                path = "graph"
            else:
                path = "vector"

            v_score = next((s for i, s in vector_candidates if i == mid), None)
            g_score = next((s for i, s in graph_candidates if i == mid), None)

            results.append(
                SearchResult(
                    memory=mem,
                    score=final_scores[mid],
                    vector_score=v_score,
                    graph_score=g_score,
                    retrieval_path=path,
                )
            )

        # 10. Update access metadata for returned memories
        now = datetime.utcnow()
        for r in results:
            self._vector_store.update_metadata(
                r.memory.id,
                access_count=r.memory.access_count + 1,
                last_accessed_at=now,
            )

        return results
