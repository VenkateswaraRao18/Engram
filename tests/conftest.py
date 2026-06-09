from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from engram.embeddings.base import Embedder
from engram.extraction.base import LLMProvider


class FakeLLM:
    def __init__(self, response_json: dict = None):
        self._response = response_json

    def complete(self, system: str, user: str) -> str:
        if self._response:
            return json.dumps(self._response)
        return json.dumps(
            {
                "memories": [
                    {
                        "content": "User lives in Tampa",
                        "memory_type": "semantic",
                        "importance": 0.85,
                    }
                ],
                "entities": [
                    {"name": "User", "entity_type": "PERSON", "aliases": []},
                    {"name": "Tampa", "entity_type": "PLACE", "aliases": []},
                ],
                "relations": [
                    {
                        "source_entity_name": "User",
                        "relation_type": "LIVES_IN",
                        "target_entity_name": "Tampa",
                        "confidence": 0.9,
                    }
                ],
            }
        )


class FakeEmbedder:
    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimensions).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def config():
    from engram.config import EngramConfig

    return EngramConfig()
