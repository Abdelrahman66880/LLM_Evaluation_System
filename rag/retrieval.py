"""Cached retrieval from the configured PDF collection."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from rag.indexing import build_index


class PDFRetriever:
    """Search a persistent vector index and format source-attributed passages."""

    def __init__(self, source: str | None = None, *, top_k: int = 4) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.source = source
        self.top_k = top_k
        self._store: InMemoryVectorStore | None = None

    @property
    def vector_store(self) -> InMemoryVectorStore:
        if self._store is None:
            self._store = build_index(self.source)
        return self._store

    def search(self, query: str) -> list[Document]:
        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty")
        return self.vector_store.similarity_search(query, k=self.top_k)

    def format_context(self, documents: list[Document]) -> str:
        if not documents:
            return "No relevant passages were found in the PDF collection."

        passages: list[str] = []
        for position, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "unknown source")
            paragraph = document.metadata.get("paragraph")
            chunk = document.metadata.get("chunk")
            location = str(source)
            if paragraph is not None:
                location += f" (paragraph {paragraph}"
                if chunk is not None:
                    location += f", chunk {chunk}"
                location += ")"
            passages.append(f"[{position}] Source: {location}\n{document.page_content}")
        return "\n\n".join(passages)

    def retrieve(self, query: str) -> str:
        return self.format_context(self.search(query))


@lru_cache(maxsize=1)
def get_default_retriever() -> PDFRetriever:
    """Return the process-wide retriever so the index is loaded only once."""
    return PDFRetriever()
