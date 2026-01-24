"""Document processing wrapper for paper pipeline."""

import logging
from pathlib import Path
from typing import Any

from workflows.document_processing.graph import process_document
from workflows.research.academic_lit_review.state import PaperMetadata

logger = logging.getLogger(__name__)


async def process_single_document(
    doi: str,
    source: str,
    paper: PaperMetadata,
    is_markdown: bool = False,
    use_batch_api: bool = True,
) -> dict[str, Any]:
    """Process a single document through document_processing workflow.

    Args:
        doi: Paper DOI
        source: Local file path OR markdown content (if is_markdown=True)
        paper: Paper metadata
        is_markdown: If True, source is markdown text, not a file path
        use_batch_api: If False, skip batch API for rapid iteration

    Returns:
        Processing result with es_record_id, zotero_key, etc.
    """
    try:
        if is_markdown:
            # Source is markdown content from OA HTML scrape
            logger.info(f"Processing {doi}: markdown content ({len(source)} chars)")
        else:
            # Source is a file path
            source_path = Path(source)
            if not source_path.exists():
                logger.error(f"Source file does not exist: {source}")
                return {
                    "doi": doi,
                    "success": False,
                    "zotero_key": None,
                    "errors": [
                        {
                            "node": "process_document",
                            "error": f"File not found: {source}",
                        }
                    ],
                }

            file_size_mb = source_path.stat().st_size / (1024 * 1024)
            logger.info(f"Processing {doi}: {source} ({file_size_mb:.1f} MB)")

        extra_metadata = {
            "DOI": doi,
            "date": paper.get("publication_date", ""),
            "publicationTitle": paper.get("venue", ""),
            "abstractNote": paper.get("abstract", "")[:500]
            if paper.get("abstract")
            else "",
        }

        result = await process_document(
            source=source,
            title=paper.get("title", "Unknown"),
            item_type="journalArticle",
            extra_metadata=extra_metadata,
            use_batch_api=use_batch_api,
        )

        status = result.get("current_status", "unknown")
        errors = result.get("errors", [])
        logger.info(
            f"Document {doi} processed with status: {status}, errors: {len(errors)}"
        )
        if errors:
            for err in errors:
                logger.warning(
                    f"  {err.get('node', 'unknown')}: {err.get('error', 'no details')}"
                )

        # Extract ES record ID from store_records list (first record at compression level 0)
        store_records = result.get("store_records", [])
        es_record_id = store_records[0].get("id") if store_records else None

        # Check for content-metadata validation failure
        validation_passed = result.get("validation_passed", True)
        validation_failed = validation_passed is False
        validation_reasoning = result.get("validation_reasoning", "")

        return {
            "doi": doi,
            "success": status not in ("failed",) and not validation_failed,
            "es_record_id": es_record_id,
            "zotero_key": result.get("zotero_key"),
            "short_summary": result.get("short_summary", ""),
            "original_language": result.get("original_language", "en"),
            "errors": errors,
            "validation_failed": validation_failed,
            "validation_reasoning": validation_reasoning,
        }

    except TimeoutError as e:
        logger.error(f"Timeout processing document {doi}: {e}")
        return {
            "doi": doi,
            "success": False,
            "zotero_key": None,
            "errors": [{"node": "process_document", "error": f"Timeout: {e}"}],
        }
    except Exception as e:
        logger.error(f"Failed to process document {doi}: {type(e).__name__}: {e}")
        return {
            "doi": doi,
            "success": False,
            "zotero_key": None,
            "errors": [
                {"node": "process_document", "error": f"{type(e).__name__}: {e}"}
            ],
        }


async def process_multiple_documents(
    documents: list[tuple[str, str, PaperMetadata]],
    use_batch_api: bool = True,
) -> list[dict[str, Any]]:
    """Process multiple documents using centralized batch processing.

    Uses process_documents_batch for concurrency control, then transforms
    results to the paper pipeline format.

    Args:
        documents: List of (doi, markdown_text, paper_metadata) tuples
        use_batch_api: Passed to each process_document call

    Returns:
        List of processing results in same format as process_single_document
    """
    if not documents:
        return []

    logger.info(f"Processing {len(documents)} documents via process_documents_batch")

    # Transform input to process_documents_batch format
    doc_configs = []
    for doi, markdown_text, paper in documents:
        doc_configs.append({
            "source": markdown_text,
            "title": paper.get("title", "Unknown"),
            "item_type": "journalArticle",
            "extra_metadata": {
                "DOI": doi,
                "date": paper.get("publication_date", ""),
                "publicationTitle": paper.get("venue", ""),
                "abstractNote": paper.get("abstract", "")[:500] if paper.get("abstract") else "",
            },
            "use_batch_api": use_batch_api,
        })

    # Use centralized batch processing (handles concurrency)
    from workflows.document_processing.graph import process_documents_batch
    raw_results = await process_documents_batch(doc_configs)

    # Transform output to paper pipeline format
    results = []
    for (doi, _, _), raw in zip(documents, raw_results):
        store_records = raw.get("store_records", [])
        validation_passed = raw.get("validation_passed", True)
        validation_failed = validation_passed is False
        results.append({
            "doi": doi,
            "success": raw.get("current_status") not in ("failed",) and not validation_failed,
            "es_record_id": store_records[0].get("id") if store_records else None,
            "zotero_key": raw.get("zotero_key"),
            "short_summary": raw.get("short_summary", ""),
            "original_language": raw.get("original_language", "en"),
            "errors": raw.get("errors", []),
            "validation_failed": validation_failed,
            "validation_reasoning": raw.get("validation_reasoning", ""),
        })

    succeeded = sum(1 for r in results if r.get("success"))
    logger.info(f"Batch processing complete: {succeeded}/{len(results)} succeeded")

    return results
