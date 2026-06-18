from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

import os

from engram import Engram, EngramConfig

from .base import BaseAdapter


class VectorOnlyAdapter(BaseAdapter):
    """
    Ablation: LLM extraction + vector retrieval, but NO temporal supersession.
    Both the old and new memories coexist — the LLM must figure out which is latest.
    This isolates the value of Engram's temporal supersession step.
    """

    name = "VectorOnly (no supersession)"

    def __init__(self) -> None:
        key = BaseAdapter.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if key:
            config = EngramConfig(
                db_path=":memory:",
                llm_provider="gemini",
                llm_model="gemini-2.5-flash",
                gemini_api_key=key,
            )
        else:
            config = EngramConfig(db_path=":memory:")
        self._client = Engram(config)
        # Disable temporal supersession by making find_conflicting a no-op
        self._client._graph_store.find_conflicting = lambda relation: []

    def add_session(self, messages: list[dict], user_id: str, session_id: str) -> None:
        self._client.add(messages, user_id=user_id)

    def answer(self, question: str, user_id: str) -> str:
        context = self._client.get_context(question, user_id=user_id, max_tokens=600)
        if not context:
            return "I don't know."
        return self._generate(question, context)

    def reset(self, user_id: str) -> None:
        self._client.forget_user(user_id)
