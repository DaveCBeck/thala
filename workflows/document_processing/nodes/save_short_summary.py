"""
Save short summary as StoreRecord node.
"""

import logging
from typing import Any
from uuid import UUID, uuid4

from core.stores.schema import SourceType, StoreRecord
from langchain_tools.base import get_store_manager
from workflows.document_processing.state import DocumentProcessingState, StoreRecordRef

logger = logging.getLogger(__name__)


async def save_short_summary(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Create StoreRecord for the short summary.

    Creates a compression_level=1 record with source_ids=[original_record_id],
    generates embedding, and saves to Elasticsearch.

    Returns updated store_records list and current_status.
    """
    try:
        short_summary = state.get("short_summary")
        if not short_summary:
            logger.error("No short_summary in state")
            return {
                "current_status": "save_summary_failed",
                "errors": [{"node": "save_short_summary", "error": "No summary to save"}],
            }

        # Find the original record (compression_level=0)
        store_records = state.get("store_records", [])
        original_record = None
        for ref in store_records:
            if ref.get("compression_level") == 0:
                original_record = ref
                break

        if not original_record:
            logger.error("No original record found in state")
            return {
                "current_status": "save_summary_failed",
                "errors": [{"node": "save_short_summary", "error": "No original record found"}],
            }

        # Create summary record
        record_id = uuid4()
        zotero_key = state.get("zotero_key")

        # Convert original record ID to UUID if it's a string
        original_id = original_record["id"]
        if isinstance(original_id, str):
            original_id = UUID(original_id)

        record = StoreRecord(
            id=record_id,
            source_type=SourceType.INTERNAL,
            zotero_key=zotero_key,
            content=short_summary,
            compression_level=1,
            source_ids=[original_id],
            metadata={
                "type": "short_summary",
                "word_count": len(short_summary.split()),
            },
        )

        # Generate embedding and save to stores
        store_manager = get_store_manager()

        # Generate embedding
        embedding = await store_manager.embedding.embed(short_summary)
        record.embedding_model = store_manager.embedding.model

        # Save to Elasticsearch
        await store_manager.es_stores.store.create(record)
        logger.info(f"Saved short summary record: {record_id}")

        # Create reference for state
        ref = StoreRecordRef(
            id=str(record_id),
            compression_level=1,
            content_preview=short_summary[:100],
        )

        return {
            "store_records": [ref],
            "current_status": "save_summary_complete",
        }

    except Exception as e:
        logger.error(f"Failed to save short summary: {e}")
        return {
            "current_status": "save_summary_failed",
            "errors": [{"node": "save_short_summary", "error": str(e)}],
        }
