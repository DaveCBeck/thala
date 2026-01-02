"""
Book download and processing node.

Downloads PDFs and converts them via Marker, then generates
theme-relevant summaries using Sonnet.
"""

import asyncio
import logging
from typing import Any

from workflows.research.subgraphs.researcher_base import fetch_pdf_via_marker
from workflows.research.subgraphs.book_finding.state import BookResult
from workflows.research.subgraphs.book_finding.prompts import SUMMARY_PROMPT
from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)

# Processing configuration
MAX_CONCURRENT_DOWNLOADS = 3
MAX_CONTENT_FOR_SUMMARY = 50000  # Characters to use for summary generation


async def _process_single_book(
    book: BookResult,
    theme: str,
) -> tuple[BookResult | None, str | None]:
    """Download, process via Marker, and summarize a single book.

    Args:
        book: BookResult to process
        theme: Theme for context-aware summarization

    Returns:
        Tuple of (updated BookResult with summary, None) on success,
        or (None, title) on failure
    """
    try:
        # Only process PDFs for now
        if book["format"].lower() != "pdf":
            logger.info(f"Skipping non-PDF book: {book['title']} (format: {book['format']})")
            return None, book["title"]

        # Download and convert via Marker
        logger.info(f"Processing book: {book['title']}")
        content = await fetch_pdf_via_marker(book["url"])

        if not content:
            logger.warning(f"No content extracted from: {book['title']}")
            return None, book["title"]

        # Truncate content for summary generation
        content_truncated = content[:MAX_CONTENT_FOR_SUMMARY]

        # Generate theme-relevant summary using Sonnet
        llm = get_llm(ModelTier.SONNET, max_tokens=1024)
        summary_prompt = SUMMARY_PROMPT.format(
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
    the Marker service.
    """
    books = state.get("search_results", [])
    theme = state.get("input", {}).get("theme", "")

    if not books:
        logger.warning("No books to process")
        return {"processed_books": [], "processing_failed": []}

    # Process books with limited concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async def process_with_semaphore(book: BookResult):
        async with semaphore:
            return await _process_single_book(book, theme)

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
