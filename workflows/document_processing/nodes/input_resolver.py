"""
Resolve input source to markdown content.

Uses get_url() for URLs (handles PDF/HTML → markdown automatically).
Handles local file paths (PDF → Marker, text → direct read).
Handles raw markdown text input directly.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from core.scraping import get_url, GetUrlOptions
from core.scraping.pdf import process_pdf_file
from workflows.shared.text_utils import chunk_by_headings, count_words, estimate_pages

logger = logging.getLogger(__name__)

# Max filename length (leave room for extension and timestamp)
MAX_FILENAME_BASE = 100

# File extensions that indicate PDF content
PDF_EXTENSIONS = {".pdf"}

# File extensions that can be read as text/markdown directly
TEXT_EXTENSIONS = {".md", ".txt", ".markdown", ".rst", ".html", ".htm"}


async def _resolve_local_file(
    source_path: Path,
    input_data: dict,
    marker_input_dir: Path,
) -> dict:
    """Resolve a local file to markdown content.

    Args:
        source_path: Path to the local file
        input_data: Input configuration dict
        marker_input_dir: Directory for saving processed output

    Returns:
        State update dict with processing_result
    """
    suffix = source_path.suffix.lower()

    if suffix in PDF_EXTENSIONS:
        # PDF file: Convert via Marker
        logger.info(f"Processing local PDF: {source_path.name} ({source_path.stat().st_size / 1024 / 1024:.1f} MB)")

        try:
            markdown = await process_pdf_file(
                str(source_path),
                quality="balanced",
                langs=input_data.get("langs", ["English"]),
            )
            ocr_method = "marker:local_pdf"
        except Exception as e:
            logger.error(f"Marker PDF processing failed: {e}")
            # Return minimal result so workflow can continue with metadata
            markdown = f"[PDF processing failed: {source_path.name}]"
            ocr_method = "marker:failed"

    elif suffix in TEXT_EXTENSIONS:
        # Text file: Read directly
        logger.info(f"Reading local text file: {source_path.name}")
        markdown = source_path.read_text(encoding="utf-8")
        ocr_method = "direct_read"

    else:
        # Unknown file type: Try to read as text
        logger.warning(f"Unknown file type '{suffix}', attempting to read as text: {source_path.name}")
        try:
            markdown = source_path.read_text(encoding="utf-8")
            ocr_method = "direct_read:unknown_type"
        except UnicodeDecodeError:
            logger.error(f"Cannot read file as text: {source_path.name}")
            markdown = f"[Cannot read file: {source_path.name}]"
            ocr_method = "failed:binary"

    # Save markdown for downstream nodes
    resolved_path = str(marker_input_dir / f"{source_path.stem}.md")
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    # Chunk by headings
    chunks = chunk_by_headings(markdown, max_chunk_size=2000)

    word_count = count_words(markdown)
    processing_result = {
        "markdown": markdown,
        "chunks": chunks,
        "page_count": estimate_pages(markdown),
        "word_count": word_count,
        "ocr_method": ocr_method,
    }

    logger.info(f"Local file resolved: {len(markdown)} chars, {word_count} words")

    return {
        "source_type": "local_file",
        "resolved_path": resolved_path,
        "processing_result": processing_result,
        "needs_tenth_summary": word_count > 2000,
        "current_status": "input_resolved",
    }


async def resolve_input(state: dict) -> dict:
    """
    Resolve input source to markdown content.

    Handles:
    - URLs: Uses get_url() which handles PDF/HTML → markdown conversion
    - Local file paths: PDF → Marker, text files → direct read
    - Markdown text: Direct processing

    Returns processing_result with markdown, chunks, and metrics.
    """
    input_data = state["input"]
    source = input_data["source"]

    marker_input_dir = Path(os.getenv("MARKER_INPUT_DIR", "/data/input"))
    marker_input_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Resolving input source: {source[:100]}...")

    # Check if source is a local file path
    source_path = Path(source)
    if source_path.exists() and source_path.is_file():
        return await _resolve_local_file(source_path, input_data, marker_input_dir)

    # Determine source type for URLs vs markdown text
    parsed_url = urlparse(source)

    if parsed_url.scheme in ("http", "https"):
        # URL: Use get_url() which handles PDF/HTML → markdown
        source_type = "url"

        result = await get_url(
            source,
            GetUrlOptions(
                pdf_quality="balanced",  # Fixed quality for PDF OCR
                pdf_langs=input_data.get("langs", ["English"]),
                detect_academic=False,
                allow_retrieve_academic=False,
            ),
        )

        markdown = result.content

        # Save markdown for downstream nodes
        filename = Path(parsed_url.path).stem or "downloaded"
        resolved_path = str(marker_input_dir / f"{filename}.md")
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        # Chunk by headings
        chunks = chunk_by_headings(markdown, max_chunk_size=2000)

        word_count = count_words(markdown)
        processing_result = {
            "markdown": markdown,
            "chunks": chunks,
            "page_count": estimate_pages(markdown),
            "word_count": word_count,
            "ocr_method": f"get_url:{result.provider or 'unknown'}",
        }

        logger.info(f"URL resolved: {len(markdown)} chars, {word_count} words")

        return {
            "source_type": source_type,
            "resolved_path": resolved_path,
            "processing_result": processing_result,
            "needs_tenth_summary": word_count > 2000,
            "current_status": "input_resolved",
        }

    else:
        # Markdown text input
        source_type = "markdown_text"
        markdown = source

        # Write to file for potential downstream use
        title = input_data.get("title", "input")
        safe_title = re.sub(r"[^\w\s-]", "", title).replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(safe_title) > MAX_FILENAME_BASE:
            safe_title = safe_title[:MAX_FILENAME_BASE]
        filename = f"{safe_title}_{timestamp}.md"
        resolved_path = str(marker_input_dir / filename)

        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        # Chunk by headings
        chunks = chunk_by_headings(markdown, max_chunk_size=2000)

        word_count = count_words(markdown)
        processing_result = {
            "markdown": markdown,
            "chunks": chunks,
            "page_count": estimate_pages(markdown),
            "word_count": word_count,
            "ocr_method": "n/a",
        }

        logger.info(f"Markdown text resolved: {len(markdown)} chars, {word_count} words")

        return {
            "source_type": source_type,
            "resolved_path": resolved_path,
            "processing_result": processing_result,
            "needs_tenth_summary": word_count > 2000,
            "current_status": "input_resolved",
        }
