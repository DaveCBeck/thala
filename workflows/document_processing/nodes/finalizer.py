"""
Final workflow step for document processing.
"""

import logging
from datetime import datetime

from langsmith import traceable

from workflows.document_processing.state import DocumentProcessingState

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="DocProcessingFinalize")
async def finalize(state: DocumentProcessingState) -> dict:
    """
    Final workflow step.

    Sets completed_at timestamp, updates status, and logs summary.
    """
    try:
        errors = state.get("errors", [])
        zotero_key = state.get("zotero_key")
        processing_result = state.get("processing_result")

        if errors:
            logger.warning(f"Workflow completed with {len(errors)} errors")
            for error in errors:
                logger.warning(f"  {error.get('node')}: {error.get('error')}")
        else:
            logger.info("Workflow completed successfully")

        if processing_result:
            page_count = processing_result.get("page_count", 0)
            word_count = processing_result.get("word_count", 0)
            logger.info(f"Processed {page_count} pages, {word_count} words")

        if zotero_key:
            logger.info(f"Zotero item: {zotero_key}")

        store_records = state.get("store_records", [])
        logger.info(f"Created {len(store_records)} store records")

        return {
            "completed_at": datetime.utcnow(),
            "current_status": "completed",
        }

    except Exception as e:
        logger.error(f"Finalize failed: {e}")
        return {
            "completed_at": datetime.utcnow(),
            "current_status": "completed_with_errors",
            "errors": [{"node": "finalizer", "error": str(e)}],
        }
