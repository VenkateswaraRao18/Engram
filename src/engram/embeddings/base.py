from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...
