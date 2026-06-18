from __future__ import annotations

import os
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """Common interface for all memory system adapters."""

    name: str = "base"

    # Set at runtime by the benchmark runner
    gemini_api_key: str = ""

    @abstractmethod
    def add_session(self, messages: list[dict], user_id: str, session_id: str) -> None:
        """Ingest one conversation session into the memory system."""
        ...

    @abstractmethod
    def answer(self, question: str, user_id: str) -> str:
        """Return an answer string given a question and the stored memory."""
        ...

    def reset(self, user_id: str) -> None:
        """Remove all stored memories for a user. Override if needed."""

    def _generate(self, question: str, context: str) -> str:
        prompt = (
            "You are a helpful assistant that answers questions about a user "
            "based only on the provided memory context. "
            "When facts conflict, trust the most specific or most recent one.\n\n"
            f"Memory context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer in one short sentence using the information above. "
            "If the answer is genuinely absent from the context, say: I don't know."
        )
        key = BaseAdapter.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if key:
            return self._generate_gemini(prompt, key)
        return self._generate_ollama(prompt)

    def _generate_gemini(self, prompt: str, api_key: str) -> str:
        import time
        import google.genai as genai
        import google.genai.types as genai_types

        client = genai.Client(api_key=api_key)
        models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]
        for attempt, model in enumerate(models_to_try):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(temperature=0),
                )
                return response.text.strip()
            except Exception as e:
                err = str(e)
                if "503" in err or "UNAVAILABLE" in err:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                raise
        return "I don't know."

    def _generate_ollama(self, prompt: str) -> str:
        import ollama

        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        return response.message.content.strip()
