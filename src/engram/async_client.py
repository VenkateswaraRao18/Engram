from __future__ import annotations

import asyncio
import os
from typing import Optional

from .client import Engram
from .config import EngramConfig
from .models import Memory, SearchResult


class AsyncEngram:
    """Async wrapper around Engram.

    All blocking operations (LLM calls, embeddings, SQLite, Neo4j) run in a
    thread-pool executor so callers can await without blocking the event loop.
    """

    def __init__(self, config: EngramConfig):
        self._sync = Engram(config)

    @classmethod
    def local(cls, path: str = "./memdb") -> AsyncEngram:
        os.makedirs(path, exist_ok=True)
        instance = object.__new__(cls)
        instance._sync = Engram.local(path)
        return instance

    async def add(self, messages: list[dict], user_id: str) -> list[Memory]:
        return await asyncio.to_thread(self._sync.add, messages, user_id)

    async def search(self, query: str, user_id: str, k: int = 5) -> list[SearchResult]:
        return await asyncio.to_thread(self._sync.search, query, user_id, k)

    async def get_context(self, query: str, user_id: str, max_tokens: int = 800) -> str:
        return await asyncio.to_thread(self._sync.get_context, query, user_id, max_tokens)

    async def forget(self, memory_id: str) -> None:
        await asyncio.to_thread(self._sync.forget, memory_id)

    async def forget_user(self, user_id: str) -> None:
        await asyncio.to_thread(self._sync.forget_user, user_id)

    async def stats(self, user_id: str) -> dict:
        return await asyncio.to_thread(self._sync.stats, user_id)
