"""Document processing wrapper for book finding pipeline.

Wraps the document_processing workflow to handle books found via book_search.
Creates Zotero entries, 10:1 summaries, and store records.
"""

import logging
from pathlib import Path
from typing import Any

from workflows.document_processing.graph import process_document

logger = logging.getLogger(__name__)


async def process_book_document(
    file_path: str,
    title: str,
    authors: str,
    md5: str,
    use_batch_api: bool = True,
) -> dict[str, Any]:
    """Process a downloaded book through document_processing workflow.

    Args:
        file_path: Local path to the downloaded PDF
        title: Book title
        authors: Book authors (comma-separated string)
        md5: MD5 hash of the book (used as identifier)
        use_batch_api: If False, skip batch API for rapid iteration

    Returns:
        Processing result with:
        - success: bool
        - zotero_key: str | None
        - short_summary: str (content_summary for backward compatibility)
        - tenth_summary: str | None (10:1 summary in original language)
        - tenth_summary_english: str | None (10:1 summary in English)
        - original_language: str
        - store_records: list[dict] (all store records created)
        - errors: list[dict]
    """
    try:
        source_path = Path(file_path)
        if not source_path.exists():
            logger.error(f"Source file does not exist: {file_path}")
            return {
                "success": False,
                "zotero_key": None,
                "short_summary": None,
                "tenth_summary": None,
                "tenth_summary_english": None,
                "original_language": None,
                "store_records": [],
                "errors": [
                    {
                        "node": "process_book_document",
                        "error": f"File not found: {file_path}",
                    }
                ],
            }

        file_size_mb = source_path.stat().st_size / (1024 * 1024)
        logger.info(f"Processing book '{title}': {file_path} ({file_size_mb:.1f} MB)")

        # Prepare metadata - books are treated as "book" item type
        extra_metadata = {
            "md5": md5,
            "creators": [
                {"creatorType": "author", "name": name.strip()}
                for name in authors.split(",")
                if name.strip()
            ],
        }

        result = await process_document(
            source=file_path,
            title=title,
            item_type="book",
            extra_metadata=extra_metadata,
            use_batch_api=use_batch_api,
        )

        status = result.get("current_status", "unknown")
        errors = result.get("errors", [])
        logger.info(
            f"Book '{title}' processed with status: {status}, errors: {len(errors)}"
        )
        if errors:
            for err in errors:
                logger.warning(
                    f"  {err.get('node', 'unknown')}: {err.get('error', 'no details')}"
                )

        # Extract store records
        store_records = result.get("store_records", [])

        # Extract tenth summary from result
        tenth_summary = result.get("tenth_summary")
        tenth_summary_english = result.get("tenth_summary_english")

        return {
            "success": status not in ("failed",),
            "zotero_key": result.get("zotero_key"),
            "short_summary": result.get("short_summary", ""),
            "tenth_summary": tenth_summary,
            "tenth_summary_english": tenth_summary_english,
            "original_language": result.get("original_language", "en"),
            "store_records": [
                {"id": rec.get("id"), "compression_level": rec.get("compression_level")}
                for rec in store_records
            ],
            "errors": errors,
        }

    except TimeoutError as e:
        logger.error(f"Timeout processing book '{title}': {e}")
        return {
            "success": False,
            "zotero_key": None,
            "short_summary": None,
            "tenth_summary": None,
            "tenth_summary_english": None,
            "original_language": None,
            "store_records": [],
            "errors": [{"node": "process_book_document", "error": f"Timeout: {e}"}],
        }
    except Exception as e:
        logger.error(f"Failed to process book '{title}': {type(e).__name__}: {e}")
        return {
            "success": False,
            "zotero_key": None,
            "short_summary": None,
            "tenth_summary": None,
            "tenth_summary_english": None,
            "original_language": None,
            "store_records": [],
            "errors": [
                {"node": "process_book_document", "error": f"{type(e).__name__}: {e}"}
            ],
        }
