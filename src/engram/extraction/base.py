from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import ExtractionResult


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...


@runtime_checkable
class Extractor(Protocol):
    def extract(self, messages: list[dict], user_id: str) -> ExtractionResult:
        ...
