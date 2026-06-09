from __future__ import annotations

import os
from typing import Optional

from .config import EngramConfig
from .consolidation.engine import ConsolidationEngine
from .embeddings.ollama_embedder import OllamaEmbedder
from .extraction.llm_extractor import LLMExtractor, OllamaProvider
from .models import Memory, SearchResult
from .retrieval.hybrid import HybridRetriever
from .stores.graph.networkx_store import NetworkXGraphStore
from .stores.vector.sqlite_store import SQLiteVectorStore


class Engram:
    """High-level client for the Engram long-term memory engine."""

    def __init__(self, config: EngramConfig):
        self._config = config

        # Build providers
        llm = OllamaProvider(model=config.llm_model)
        embedder = OllamaEmbedder(model=config.embedding_model)

        # Build stores
        self._vector_store = SQLiteVectorStore(db_path=config.db_path)
        self._graph_store = NetworkXGraphStore(path=config.graph_path)

        # Build components
        self._extractor = LLMExtractor(llm=llm, window=config.extraction_window)
        self._consolidator = ConsolidationEngine(
            embedder=embedder,
            vector_store=self._vector_store,
            graph_store=self._graph_store,
            config=config,
        )
        self._retriever = HybridRetriever(
            embedder=embedder,
            vector_store=self._vector_store,
            graph_store=self._graph_store,
            config=config,
        )

    @classmethod
    def local(cls, path: str = "./memdb") -> Engram:
        """Create a Engram instance persisted to a local directory."""
        os.makedirs(path, exist_ok=True)
        return cls(
            EngramConfig(
                db_path=os.path.join(path, "memories.db"),
                graph_path=os.path.join(path, "graph.json"),
            )
        )

    def add(self, messages: list[dict], user_id: str) -> list[Memory]:
        """Extract and store memories from a list of messages."""
        extraction = self._extractor.extract(messages, user_id)
        return self._consolidator.process_extraction(extraction, user_id)

    def search(self, query: str, user_id: str, k: int = 5) -> list[SearchResult]:
        """Search for relevant memories."""
        return self._retriever.search(query, user_id, k)

    def get_context(self, query: str, user_id: str, max_tokens: int = 800) -> str:
        """Get a context string for injection into an LLM prompt."""
        results = self.search(query, user_id, k=10)
        lines = []
        token_count = 0
        for r in results:
            line = f"- [{r.memory.memory_type.value}] {r.memory.content}"
            tokens = len(line.split())
            if token_count + tokens > max_tokens:
                break
            lines.append(line)
            token_count += tokens
        return "\n".join(lines)

    def forget(self, memory_id: str) -> None:
        """Delete a specific memory by ID."""
        self._vector_store.delete([memory_id])

    def forget_user(self, user_id: str) -> None:
        """Delete all memories for a user."""
        memories = self._vector_store.get_all_valid(user_id)
        self._vector_store.delete([m.id for m in memories])

    def stats(self, user_id: str) -> dict:
        """Return statistics about stored memories."""
        return {"total_memories": self._vector_store.count(user_id)}
