"""PDF text extraction and paragraph chunking."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


def extract_paragraphs_from_pdf(file_path: str | Path) -> list[str]:
    """Extract non-empty text blocks from a PDF in reading order."""
    path = Path(file_path).expanduser()
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF file not found: {path}")

    paragraphs: list[str] = []
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            blocks = sorted(page.get_text("blocks"), key=lambda block: (block[1], block[0]))
            for block in blocks:
                text = re.sub(r"\s+", " ", block[4]).strip()
                if text:
                    paragraphs.append(text)
    return paragraphs


def pdf_to_documents(
    file_path: str | Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Convert PDF text blocks into bounded, source-attributed chunks."""
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0")

    path = Path(file_path).expanduser().resolve()
    documents: list[Document] = []
    for paragraph_number, paragraph in enumerate(extract_paragraphs_from_pdf(path)):
        start = 0
        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            if end < len(paragraph):
                boundary = paragraph.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunk = paragraph[start:end].strip()
            if chunk:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": str(path),
                            "paragraph": paragraph_number,
                            "chunk": len(documents),
                        },
                    )
                )
            if end >= len(paragraph):
                break
            start = max(end - chunk_overlap, start + 1)
    return documents
