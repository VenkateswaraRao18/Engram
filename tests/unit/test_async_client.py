from __future__ import annotations

import pytest
import pytest_asyncio

from engram.async_client import AsyncEngram
from engram.config import EngramConfig


@pytest.fixture
def async_engram(fake_llm, fake_embedder):
    """AsyncEngram wired with FakeLLM and FakeEmbedder, no real Ollama calls."""
    config = EngramConfig()
    client = AsyncEngram(config)
    client._sync._extractor.llm = fake_llm
    client._sync._consolidator._embedder = fake_embedder
    client._sync._retriever._embedder = fake_embedder
    return client


@pytest.mark.asyncio
async def test_async_add_returns_memories(async_engram):
    memories = await async_engram.add(
        messages=[{"role": "user", "content": "I live in Tampa"}],
        user_id="u1",
    )
    assert isinstance(memories, list)
    assert len(memories) >= 1


@pytest.mark.asyncio
async def test_async_search_returns_results(async_engram):
    await async_engram.add(
        messages=[{"role": "user", "content": "I live in Tampa"}],
        user_id="u1",
    )
    results = await async_engram.search("Where do I live?", user_id="u1", k=5)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_async_get_context_returns_string(async_engram):
    await async_engram.add(
        messages=[{"role": "user", "content": "I live in Tampa"}],
        user_id="u1",
    )
    ctx = await async_engram.get_context("location", user_id="u1")
    assert isinstance(ctx, str)


@pytest.mark.asyncio
async def test_async_stats(async_engram):
    await async_engram.add(
        messages=[{"role": "user", "content": "I live in Tampa"}],
        user_id="u1",
    )
    stats = await async_engram.stats("u1")
    assert "total_memories" in stats
    assert stats["total_memories"] >= 1


@pytest.mark.asyncio
async def test_async_forget(async_engram):
    memories = await async_engram.add(
        messages=[{"role": "user", "content": "I live in Tampa"}],
        user_id="u1",
    )
    before = await async_engram.stats("u1")
    if memories:
        await async_engram.forget(memories[0].id)
    after = await async_engram.stats("u1")
    assert after["total_memories"] <= before["total_memories"]


@pytest.mark.asyncio
async def test_async_forget_user(async_engram):
    await async_engram.add(
        messages=[{"role": "user", "content": "I live in Tampa"}],
        user_id="wipe_me",
    )
    await async_engram.forget_user("wipe_me")
    stats = await async_engram.stats("wipe_me")
    assert stats["total_memories"] == 0


@pytest.mark.asyncio
async def test_async_sequential_adds(async_engram):
    r1 = await async_engram.add([{"role": "user", "content": "I live in Tampa"}], "u_seq")
    r2 = await async_engram.add([{"role": "user", "content": "I like coffee"}], "u_seq")
    assert isinstance(r1, list)
    assert isinstance(r2, list)
