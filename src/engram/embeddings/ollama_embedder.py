from __future__ import annotations

import ollama


class OllamaEmbedder:
    """Embedder that uses Ollama's embedding API."""

    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = ollama.embeddings(model=self.model, prompt=text)
        return response["embedding"]
