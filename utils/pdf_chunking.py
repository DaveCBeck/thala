"""PDF chunking utilities for handling large PDFs.

Provides functions to split large PDFs into smaller chunks for processing,
and reassemble the resulting markdown outputs.
"""

import io
import logging
import re
from typing import BinaryIO

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def get_page_count(content: bytes | BinaryIO) -> int:
    """Get the number of pages in a PDF.

    Args:
        content: PDF content as bytes or file-like object

    Returns:
        Number of pages in the PDF
    """
    if isinstance(content, bytes):
        content = io.BytesIO(content)
    reader = PdfReader(content)
    return len(reader.pages)


def should_chunk_pdf(content: bytes, page_threshold: int = 100) -> bool:
    """Determine if a PDF should be chunked based on page count.

    Args:
        content: PDF content as bytes
        page_threshold: Minimum pages before chunking is triggered

    Returns:
        True if PDF should be chunked, False otherwise
    """
    try:
        page_count = get_page_count(content)
        return page_count >= page_threshold
    except Exception as e:
        logger.warning(f"Could not determine page count, skipping chunking: {e}")
        return False


def split_pdf_by_pages(
    content: bytes,
    chunk_size: int = 100,
) -> list[tuple[bytes, tuple[int, int]]]:
    """Split a PDF into chunks of N pages each.

    Args:
        content: PDF content as bytes
        chunk_size: Maximum pages per chunk

    Returns:
        List of (chunk_bytes, (start_page, end_page)) tuples.
        Page numbers are 1-indexed for human readability.
    """
    reader = PdfReader(io.BytesIO(content))
    total_pages = len(reader.pages)

    if total_pages <= chunk_size:
        # No need to split
        return [(content, (1, total_pages))]

    chunks = []
    for start_idx in range(0, total_pages, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pages)

        writer = PdfWriter()
        for page_idx in range(start_idx, end_idx):
            writer.add_page(reader.pages[page_idx])

        # Write chunk to bytes
        chunk_buffer = io.BytesIO()
        writer.write(chunk_buffer)
        chunk_bytes = chunk_buffer.getvalue()

        # Page numbers are 1-indexed for readability
        page_range = (start_idx + 1, end_idx)
        chunks.append((chunk_bytes, page_range))

        logger.debug(f"Created chunk: pages {page_range[0]}-{page_range[1]}")

    logger.info(f"Split PDF ({total_pages} pages) into {len(chunks)} chunks of ~{chunk_size} pages")
    return chunks


def assemble_markdown_chunks(
    chunks: list[str],
    page_ranges: list[tuple[int, int]],
) -> str:
    """Reassemble markdown chunks into a single document.

    Args:
        chunks: List of markdown strings from each chunk
        page_ranges: List of (start_page, end_page) tuples for each chunk

    Returns:
        Combined markdown with page range annotations
    """
    if len(chunks) != len(page_ranges):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(page_ranges)} page ranges"
        )

    if len(chunks) == 1:
        # Single chunk, no annotation needed
        return chunks[0]

    assembled_parts = []
    seen_h1 = False

    for i, (markdown, (start_page, end_page)) in enumerate(zip(chunks, page_ranges)):
        # Add page range annotation
        annotation = f"\n\n<!-- Pages {start_page}-{end_page} -->\n\n"

        # Normalize heading levels after first chunk to avoid duplicate H1s
        if i > 0:
            markdown = _demote_h1_headings(markdown, seen_h1)

        # Track if we've seen an H1
        if re.search(r"^# [^#]", markdown, re.MULTILINE):
            seen_h1 = True

        assembled_parts.append(annotation + markdown.strip())

    return "\n\n".join(assembled_parts)


def _demote_h1_headings(markdown: str, demote: bool) -> str:
    """Demote H1 headings to H2 if we've already seen an H1.

    This prevents multiple top-level headings in the assembled document.

    Args:
        markdown: Markdown content
        demote: Whether to demote H1 headings

    Returns:
        Markdown with potentially demoted headings
    """
    if not demote:
        return markdown

    # Only demote H1 headings (single #) to H2
    # Be careful not to demote ## or ### etc.
    return re.sub(r"^# ([^#])", r"## \1", markdown, flags=re.MULTILINE)
