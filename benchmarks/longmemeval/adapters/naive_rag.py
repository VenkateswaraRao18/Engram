from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from engram.embeddings.ollama_embedder import OllamaEmbedder
from engram.models import Memory, MemoryType
from engram.stores.vector.sqlite_store import SQLiteVectorStore

from .base import BaseAdapter


class NaiveRAGAdapter(BaseAdapter):
    """
    Baseline: no LLM extraction, no knowledge graph, no temporal supersession.
    Each user turn is embedded and stored as a raw chunk. Retrieval is pure
    vector similarity. This is the simplest possible RAG memory system.
    """

    name = "NaiveRAG (baseline)"

    def __init__(self) -> None:
        self._embedder = OllamaEmbedder()
        self._store = SQLiteVectorStore(db_path=":memory:")

    def add_session(self, messages: list[dict], user_id: str, session_id: str) -> None:
        for msg in messages:
            if msg["role"] != "user":
                continue
            content = msg["content"].strip()
            if not content:
                continue
            embedding = self._embedder.embed(content)
            memory = Memory(
                user_id=user_id,
                content=content,
                memory_type=MemoryType.SEMANTIC,
                embedding=embedding,
            )
            self._store.add([memory])

    def answer(self, question: str, user_id: str) -> str:
        query_emb = self._embedder.embed(question)
        hits = self._store.search(user_id, query_emb, k=5, filter_valid=True)
        if not hits:
            return "I don't know."
        chunks = []
        for mem_id, _score in hits:
            m = self._store.get(mem_id)
            if m:
                chunks.append(m.content)
        context = "\n".join(chunks)
        return self._generate(question, context)

    def reset(self, user_id: str) -> None:
        memories = self._store.get_all_valid(user_id)
        self._store.delete([m.id for m in memories])
