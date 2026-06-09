from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ...models import Memory


@runtime_checkable
class VectorStore(Protocol):
    def add(self, memories: list[Memory]) -> None:
        ...

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        k: int = 20,
        filter_valid: bool = True,
    ) -> list[tuple[str, float]]:
        ...

    def delete(self, memory_ids: list[str]) -> None:
        ...

    def update_metadata(self, memory_id: str, **fields) -> None:
        ...

    def get(self, memory_id: str) -> Optional[Memory]:
        ...

    def get_all_valid(self, user_id: str) -> list[Memory]:
        ...

    def count(self, user_id: str) -> int:
        ...
