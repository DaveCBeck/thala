"""
Resolve input source to markdown content.

Uses get_url() for URLs (handles PDF/HTML → markdown automatically).
Handles raw markdown text input directly.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from core.scraping import get_url, GetUrlOptions
from workflows.shared.text_utils import chunk_by_headings, count_words, estimate_pages

logger = logging.getLogger(__name__)

# Max filename length (leave room for extension and timestamp)
MAX_FILENAME_BASE = 100


async def resolve_input(state: dict) -> dict:
    """
    Resolve input source to markdown content.

    Handles:
    - URLs: Uses get_url() which handles PDF/HTML → markdown conversion
    - Markdown text: Direct processing

    Returns processing_result with markdown, chunks, and metrics.
    """
    input_data = state["input"]
    source = input_data["source"]

    marker_input_dir = Path(os.getenv("MARKER_INPUT_DIR", "/data/input"))
    marker_input_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Resolving input source: {source[:100]}...")

    # Determine source type
    parsed_url = urlparse(source)

    if parsed_url.scheme in ("http", "https"):
        # URL: Use get_url() which handles PDF/HTML → markdown
        source_type = "url"

        result = await get_url(
            source,
            GetUrlOptions(
                pdf_quality=input_data.get("quality", "balanced"),
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
