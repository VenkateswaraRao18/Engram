from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EngramConfig(BaseModel):
    llm_model: str = "gemini-2.5-flash"
    llm_provider: str = "ollama"    # "ollama" | "gemini"
    gemini_api_key: Optional[str] = None
    embedding_provider: str = "ollama"   # "ollama" | "sentence-transformers"
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
    vector_store: str = "sqlite"    # "sqlite" | "faiss"
    graph_store: str = "networkx"   # "networkx" | "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: Optional[str] = None
