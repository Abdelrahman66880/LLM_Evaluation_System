"""Create an in-memory vector index from a PDF using OpenRouter embeddings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

try:  # Support both ``python -m RAG1.indexing`` and running from RAG1/.
    from .file_reader import extract_paragraphs_from_pdf
except ImportError:  # pragma: no cover - used for direct script execution
    from file_reader import extract_paragraphs_from_pdf


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


class OpenRouterEmbeddings(Embeddings):
    """LangChain embeddings client backed by OpenRouter's embeddings API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = OPENROUTER_EMBEDDINGS_URL,
        batch_size: int = 64,
        timeout: float = 60,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add your OpenRouter key to the environment or .env file."
            )
        self.model = model or os.getenv("OPENROUTER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = Request(
            self.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "LLM Evaluation System"),
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter embeddings request failed ({exc.code}): {message}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc

        data = result.get("data")
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


def index_pdf(
    file_path: str | Path,
    *,
    embedding_model: str | None = None,
    batch_size: int = 64,
) -> InMemoryVectorStore:
    """Extract paragraphs from a PDF or all PDFs in a directory and index them.

    Query the returned store with LangChain's ``similarity_search``. The index
    is in memory only, so it must be rebuilt after restarting the application.
    """
    source = Path(file_path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file or directory, got: {source}")
        pdf_files = [source]
    elif source.is_dir():
        pdf_files = sorted(
            item for item in source.rglob("*") if item.is_file() and item.suffix.lower() == ".pdf"
        )
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {source}")
    else:
        raise FileNotFoundError(f"PDF file or directory does not exist: {source}")

    texts: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    for pdf_file in pdf_files:
        paragraphs = extract_paragraphs_from_pdf(str(pdf_file))
        texts.extend(paragraphs)
        metadatas.extend(
            {"source": str(pdf_file), "paragraph": number}
            for number in range(len(paragraphs))
        )
    if not texts:
        raise ValueError(f"No text paragraphs were found in PDF source: {source}")

    embeddings = OpenRouterEmbeddings(model=embedding_model, batch_size=batch_size)
    store = InMemoryVectorStore(embeddings)
    store.add_texts(texts, metadatas=metadatas)
    return store

