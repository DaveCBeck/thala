"""Document processing wrapper for paper pipeline."""

import logging
from pathlib import Path
from typing import Any

from workflows.document_processing.graph import process_document
from workflows.research.subgraphs.academic_lit_review.state import PaperMetadata

logger = logging.getLogger(__name__)


async def process_single_document(
    doi: str,
    local_path: str,
    paper: PaperMetadata,
) -> dict[str, Any]:
    """Process a single document through document_processing workflow.

    Args:
        doi: Paper DOI
        local_path: Local file path
        paper: Paper metadata

    Returns:
        Processing result with es_record_id, zotero_key, etc.
    """
    try:
        source_path = Path(local_path)
        if not source_path.exists():
            logger.error(f"Source file does not exist: {local_path}")
            return {
                "doi": doi,
                "success": False,
                "errors": [{"node": "process_document", "error": f"File not found: {local_path}"}],
            }

        file_size_mb = source_path.stat().st_size / (1024 * 1024)
        logger.info(f"Processing {doi}: {local_path} ({file_size_mb:.1f} MB)")

        extra_metadata = {
            "DOI": doi,
            "date": paper.get("publication_date", ""),
            "publicationTitle": paper.get("venue", ""),
            "abstractNote": paper.get("abstract", "")[:500] if paper.get("abstract") else "",
        }

        result = await process_document(
            source=local_path,
            title=paper.get("title", "Unknown"),
            item_type="journalArticle",
            quality="fast",
            extra_metadata=extra_metadata,
        )

        status = result.get("current_status", "unknown")
        errors = result.get("errors", [])
        logger.info(f"Document {doi} processed with status: {status}, errors: {len(errors)}")
        if errors:
            for err in errors:
                logger.warning(f"  {err.get('node', 'unknown')}: {err.get('error', 'no details')}")

        return {
            "doi": doi,
            "success": status not in ("failed",),
            "es_record_id": result.get("store_record_id"),
            "zotero_key": result.get("zotero_key"),
            "short_summary": result.get("short_summary", ""),
            "errors": errors,
        }

    except TimeoutError as e:
        logger.error(f"Timeout processing document {doi}: {e}")
        return {
            "doi": doi,
            "success": False,
            "errors": [{"node": "process_document", "error": f"Timeout: {e}"}],
        }
    except Exception as e:
        logger.error(f"Failed to process document {doi}: {type(e).__name__}: {e}")
        return {
            "doi": doi,
            "success": False,
            "errors": [{"node": "process_document", "error": f"{type(e).__name__}: {e}"}],
        }
