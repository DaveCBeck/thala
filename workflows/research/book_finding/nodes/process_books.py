"""
Book download and processing node.

Downloads PDFs via retrieve-academic and processes them through
document_processing workflow to create Zotero entries and 10:1 summaries.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from core.scraping import download_pdf_by_md5
from workflows.research.book_finding.state import (
    BookResult,
    BookFindingQualitySettings,
    QUALITY_PRESETS,
)
from workflows.research.book_finding.document_processing import process_book_document

logger = logging.getLogger(__name__)


async def _process_single_book(
    book: BookResult,
    theme: str,
    quality_settings: BookFindingQualitySettings,
) -> tuple[BookResult | None, str | None]:
    """Download and process a single book through document_processing.

    Args:
        book: BookResult to process
        theme: Theme for context (logged but not used for summarization -
               document_processing handles that)
        quality_settings: Quality configuration

    Returns:
        Tuple of (updated BookResult with zotero_key and summaries, None) on success,
        or (None, title) on failure
    """
    try:
        # Only process PDFs for now
        if book["format"].lower() != "pdf":
            logger.debug(
                f"Skipping non-PDF: '{book['title']}' (format: {book['format']})"
            )
            return None, book["title"]

        # Download PDF to temporary location
        logger.debug(f"Downloading '{book['title']}'")

        # Create temp file with meaningful name
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in book["title"][:50])
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            prefix=f"book_{safe_title}_",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Download via retrieve-academic (VPN-enabled)
            local_path = await download_pdf_by_md5(
                md5=book["md5"],
                output_path=tmp_path,
                identifier=book["title"],
                timeout=120.0,
            )

            if not local_path:
                logger.warning(f"Download failed for '{book['title']}'")
                return None, book["title"]

            # Process through document_processing workflow
            logger.debug(f"Processing '{book['title']}' through document_processing")
            result = await process_book_document(
                file_path=local_path,
                title=book["title"],
                authors=book["authors"],
                md5=book["md5"],
                use_batch_api=True,  # Use batch API for cost savings
            )

            if not result["success"]:
                logger.warning(
                    f"Document processing failed for '{book['title']}': "
                    f"{result.get('errors', [])}"
                )
                return None, book["title"]

            # Create updated book with new fields from document_processing
            updated_book = BookResult(
                title=book["title"],
                authors=book["authors"],
                md5=book["md5"],
                url=book["url"],
                format=book["format"],
                size=book["size"],
                abstract=book["abstract"],
                matched_recommendation=book["matched_recommendation"],
                # Use short_summary as content_summary for backward compatibility
                content_summary=result.get("short_summary"),
                # New fields from document_processing
                zotero_key=result.get("zotero_key"),
                tenth_summary=result.get("tenth_summary"),
                tenth_summary_english=result.get("tenth_summary_english"),
                original_language=result.get("original_language"),
                store_records=result.get("store_records"),
            )

            logger.info(
                f"Successfully processed '{book['title']}' "
                f"(zotero_key={result.get('zotero_key')})"
            )
            return updated_book, None

        finally:
            # Cleanup temp file
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    except Exception as e:
        logger.error(f"Failed to process '{book['title']}': {e}")
        return None, book["title"]


async def process_books(state: dict) -> dict[str, Any]:
    """Download and process books via document_processing workflow.

    Downloads PDFs via retrieve-academic (VPN-enabled), then processes
    each through document_processing to create:
    - Zotero library entries with "processed" tag
    - 10:1 summaries stored in Elasticsearch
    - Store records for later retrieval
    """
    books = state.get("search_results", [])
    theme = state.get("input", {}).get("theme", "")
    quality_settings = state.get("quality_settings") or QUALITY_PRESETS["standard"]

    if not books:
        logger.warning("No books to process")
        return {"processed_books": [], "processing_failed": []}

    logger.info(
        f"Processing {len(books)} books with max_concurrent="
        f"{quality_settings['max_concurrent_downloads']}"
    )

    # Process books with limited concurrency based on quality settings
    max_concurrent = quality_settings["max_concurrent_downloads"]
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(book: BookResult):
        async with semaphore:
            return await _process_single_book(book, theme, quality_settings)

    tasks = [process_with_semaphore(book) for book in books]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    failed = []

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Processing task failed: {result}")
            continue

        updated_book, failed_title = result
        if updated_book:
            processed.append(updated_book)
        elif failed_title:
            failed.append(failed_title)

    # Log summary with Zotero keys
    zotero_keys = [b.get("zotero_key") for b in processed if b.get("zotero_key")]
    logger.info(
        f"Book processing complete: {len(processed)} processed, {len(failed)} failed/skipped. "
        f"Zotero keys: {zotero_keys}"
    )

    return {
        "processed_books": processed,
        "processing_failed": failed,
        "current_phase": "synthesize",
    }
