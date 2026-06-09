from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EngramConfig(BaseModel):
    llm_model: str = "llama3.1"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    db_path: str = ":memory:"
    graph_path: Optional[str] = None
    rrf_k: int = 60
    recency_weight: float = 0.3
    importance_weight: float = 0.2
    recency_half_life_days: float = 30.0
    dedup_similarity_threshold: float = 0.92
    dedup_top_k: int = 5
    extraction_window: int = 6
