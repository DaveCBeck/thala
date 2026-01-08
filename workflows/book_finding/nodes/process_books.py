"""
Book download and processing node.

Downloads PDFs and converts them via Marker, then generates
theme-relevant summaries using Sonnet.
"""

import asyncio
import logging
from typing import Any

from workflows.research.subgraphs.researcher_base import fetch_pdf_via_marker
from workflows.book_finding.state import (
    BookResult,
    BookFindingQualitySettings,
    BOOK_QUALITY_PRESETS,
)
from workflows.book_finding.prompts import get_summary_prompt
from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)


async def _process_single_book(
    book: BookResult,
    theme: str,
    quality_settings: BookFindingQualitySettings,
    summary_prompt_template: str,
) -> tuple[BookResult | None, str | None]:
    """Download, process via Marker, and summarize a single book.

    Args:
        book: BookResult to process
        theme: Theme for context-aware summarization
        quality_settings: Quality configuration for content limits and tokens
        summary_prompt_template: Prompt template for summary generation

    Returns:
        Tuple of (updated BookResult with summary, None) on success,
        or (None, title) on failure
    """
    try:
        # Only process PDFs for now
        if book["format"].lower() != "pdf":
            logger.info(f"Skipping non-PDF book: {book['title']} (format: {book['format']})")
            return None, book["title"]

        # Download via VPN and convert via Marker
        logger.info(f"Processing book: {book['title']}")
        content = await fetch_pdf_via_marker(
            md5=book["md5"],
            identifier=book["title"],
        )

        if not content:
            logger.warning(f"No content extracted from: {book['title']}")
            return None, book["title"]

        # Truncate content for summary generation based on quality settings
        max_content = quality_settings["max_content_for_summary"]
        content_truncated = content[:max_content]

        # Generate theme-relevant summary using Sonnet
        summary_tokens = quality_settings["summary_max_tokens"]
        llm = get_llm(ModelTier.SONNET, max_tokens=summary_tokens)
        summary_prompt = summary_prompt_template.format(
            theme=theme,
            title=book["title"],
            authors=book["authors"],
            content=content_truncated,
        )

        response = await llm.ainvoke([{"role": "user", "content": summary_prompt}])
        summary = response.content if isinstance(response.content, str) else str(response.content)
        summary = summary.strip()

        # Create updated book with summary
        updated_book = BookResult(
            title=book["title"],
            authors=book["authors"],
            md5=book["md5"],
            url=book["url"],
            format=book["format"],
            size=book["size"],
            abstract=book["abstract"],
            matched_recommendation=book["matched_recommendation"],
            content_summary=summary,
        )

        logger.info(f"Successfully processed: {book['title']}")
        return updated_book, None

    except Exception as e:
        logger.error(f"Failed to process book '{book['title']}': {e}")
        return None, book["title"]


async def process_books(state: dict) -> dict[str, Any]:
    """Download and process books via Marker, generate theme-relevant summaries.

    Processes books with limited concurrency to avoid overwhelming
    the Marker service. Concurrency is controlled by quality settings.
    """
    books = state.get("search_results", [])
    theme = state.get("input", {}).get("theme", "")
    quality_settings = state.get("quality_settings") or BOOK_QUALITY_PRESETS["standard"]
    language_config = state.get("language_config")

    if not books:
        logger.warning("No books to process")
        return {"processed_books": [], "processing_failed": []}

    # Get translated summary prompt if needed
    summary_prompt_template = await get_summary_prompt(language_config)

    # Process books with limited concurrency based on quality settings
    max_concurrent = quality_settings["max_concurrent_downloads"]
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(book: BookResult):
        async with semaphore:
            return await _process_single_book(book, theme, quality_settings, summary_prompt_template)

    tasks = [process_with_semaphore(book) for book in books]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    failed = []

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Processing task failed with exception: {result}")
            continue

        updated_book, failed_title = result
        if updated_book:
            processed.append(updated_book)
        elif failed_title:
            failed.append(failed_title)

    logger.info(f"Processed {len(processed)} books, {len(failed)} failed/skipped")

    return {
        "processed_books": processed,
        "processing_failed": failed,
        "current_phase": "synthesize",
    }
