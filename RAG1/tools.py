"""LangChain retrieval tools that query the index built by ``indexing.py``."""

from __future__ import annotations

import os
from pathlib import Path

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv

try:  # Support package imports and running directly from RAG1/.
    from .indexing import index_pdf
except ImportError:  # pragma: no cover - used for direct script execution
    from indexing import index_pdf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "PDFs"
last_retrieved_context: list[Document] = []
_indexed_path: Path | None = None
_vector_store: InMemoryVectorStore | None = None


def _get_vector_store() -> InMemoryVectorStore:
    """Build the index through ``indexing.index_pdf`` once and reuse it."""
    global _indexed_path, _vector_store

    load_dotenv()
    source = Path(os.getenv("RAG_PDF_PATH", str(DEFAULT_PDF_PATH))).expanduser().resolve()
    if _vector_store is None or source != _indexed_path:
        _vector_store = index_pdf(source)
        _indexed_path = source
    return _vector_store


@tool
def collecting_info(query: str) -> str:
    """Search the configured PDF knowledge base for passages relevant to a question.

    Searches ``RAG_PDF_PATH`` when set, otherwise all PDFs under the repository's
    ``PDFs`` directory. Returns up to four passages with their source locations.
    """
    global last_retrieved_context

    query = query.strip()
    if not query:
        last_retrieved_context = []
        return "Please provide a non-empty question to search the PDF knowledge base."

    results = _get_vector_store().similarity_search(query, k=4)
    last_retrieved_context = results
    if not results:
        return "No relevant passages were found in the PDF knowledge base."

    passages = []
    for number, document in enumerate(results, start=1):
        source = document.metadata.get("source", "unknown source")
        paragraph = document.metadata.get("paragraph")
        location = f"{source} (paragraph {paragraph})" if paragraph is not None else source
        passages.append(f"[{number}] Source: {location}\n{document.page_content}")
    return "\n\n".join(passages)
