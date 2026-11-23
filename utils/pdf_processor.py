"""PDF text extraction and simple page-aware chunking."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    page_number: int
    chunk_number: int


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text from each PDF page, preserving its 1-based page number."""
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError("This PDF is password-protected.") from exc
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Could not read this PDF. Please upload a valid PDF file.") from exc

    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((page_number, text))

    if not pages:
        raise ValueError(
            "No selectable text was found. This may be a scanned PDF; OCR is not included in this first version."
        )

    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    document_id: str,
    chunk_words: int = 180,
    overlap_words: int = 30,
) -> list[TextChunk]:
    """Split page text into overlapping word-based chunks."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero.")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be between 0 and chunk_words - 1.")

    chunks: list[TextChunk] = []
    step = chunk_words - overlap_words

    for page_number, page_text in pages:
        words = page_text.split()
        if not words:
            continue

        chunk_number = 0
        for start in range(0, len(words), step):
            section = words[start : start + chunk_words]
            if not section:
                break

            chunk_text = " ".join(section).strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document_id}-p{page_number}-c{chunk_number}",
                        text=chunk_text,
                        page_number=page_number,
                        chunk_number=chunk_number,
                    )
                )
                chunk_number += 1

            if start + chunk_words >= len(words):
                break

    return chunks
