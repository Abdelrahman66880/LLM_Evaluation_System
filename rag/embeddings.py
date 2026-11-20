"""OpenRouter embeddings adapter for LangChain vector stores."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from config import get_settings
from providers.openrouter import OpenRouterClient


class OpenRouterEmbeddings(Embeddings):
    """Request document and query embeddings through OpenRouter."""

    def __init__(
        self,
        *,
        model: str | None = None,
        batch_size: int = 64,
        timeout: float = 60,
    ) -> None:
        self.settings = get_settings()
        self.model = model or self.settings.embedding_model
        self.batch_size = batch_size
        self.timeout = timeout
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.client = OpenRouterClient(self.settings, timeout=timeout)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = self.client.post("embeddings", {"model": self.model, "input": texts})
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("OpenRouter returned an unexpected embeddings response")
        data.sort(key=lambda item: item.get("index", 0))
        try:
            return [[float(value) for value in item["embedding"]] for item in data]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("OpenRouter returned malformed embedding vectors") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]
