from __future__ import annotations


class OllamaEmbedder:
    """Embedder that uses Ollama's embedding API."""

    def __init__(self, model: str = "nomic-embed-text"):
        try:
            import ollama as _ollama
        except ImportError:
            raise ImportError("ollama not installed. Run: pip install ollama")
        self._ollama = _ollama
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = self._ollama.embeddings(model=self.model, prompt=text)
        return response["embedding"]
