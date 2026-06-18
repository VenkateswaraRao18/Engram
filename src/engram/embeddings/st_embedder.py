from __future__ import annotations


class SentenceTransformerEmbedder:
    """Embedder using sentence-transformers — no API key, works in Colab."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

    @property
    def dimensions(self) -> int:
        return self._model.get_sentence_embedding_dimension()
