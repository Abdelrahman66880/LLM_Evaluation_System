"""Build and persist a local vector index for one or more PDFs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.vectorstores import InMemoryVectorStore

from config import get_settings
from rag.documents import pdf_to_documents
from rag.embeddings import OpenRouterEmbeddings

logger = logging.getLogger(__name__)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def find_pdfs(source: str | Path) -> list[Path]:
    """Resolve a PDF path or recursively collect PDFs from a directory."""
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file or directory, got: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"PDF source does not exist: {path}")

    pdfs = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() == ".pdf")
    if not pdfs:
        raise ValueError(f"No PDF files found in: {path}")
    return pdfs


def _source_manifest(
    pdfs: list[Path], embedding_model: str, batch_size: int
) -> dict[str, Any]:
    return {
        "embedding_model": embedding_model,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "batch_size": batch_size,
        "files": [
            {"path": str(pdf), "size": pdf.stat().st_size, "modified_ns": pdf.stat().st_mtime_ns}
            for pdf in pdfs
        ],
    }


def build_index(
    source: str | Path | None = None,
    *,
    persist: bool = True,
    force_rebuild: bool = False,
    embedding_model: str | None = None,
    batch_size: int = 64,
) -> InMemoryVectorStore:
    """Load a valid local index or build and persist embeddings for the PDFs."""
    settings = get_settings()
    pdfs = find_pdfs(source or settings.pdf_path)
    embeddings = OpenRouterEmbeddings(model=embedding_model, batch_size=batch_size)
    manifest = _source_manifest(pdfs, embeddings.model, batch_size)
    index_path = settings.index_path
    manifest_path = index_path.with_suffix(index_path.suffix + ".meta.json")

    if persist and not force_rebuild and index_path.is_file() and manifest_path.is_file():
        try:
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached_manifest == manifest:
                return InMemoryVectorStore.load(str(index_path), embeddings)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Could not load cached vector index; rebuilding it: %s", exc)

    documents = [
        document
        for pdf in pdfs
        for document in pdf_to_documents(
            pdf, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    ]
    if not documents:
        raise ValueError(f"No readable text found in PDFs under {source or settings.pdf_path}")

    store = InMemoryVectorStore(embeddings)
    store.add_documents(documents)
    if persist:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        store.dump(str(index_path))
        temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary_manifest.replace(manifest_path)
    return store
