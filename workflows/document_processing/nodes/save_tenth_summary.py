"""
Save 10:1 summary as StoreRecord node.
"""

import logging
from typing import Any
from uuid import UUID, uuid4

from core.stores.schema import SourceType, StoreRecord
from langchain_tools.base import get_store_manager
from workflows.document_processing.state import DocumentProcessingState, StoreRecordRef

logger = logging.getLogger(__name__)


async def save_tenth_summary(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Create StoreRecord for the 10:1 summary.

    Creates a compression_level=2 record with source_ids=[original_record_id],
    generates embedding, and saves to Elasticsearch.

    Returns updated store_records list and current_status.
    """
    try:
        tenth_summary = state.get("tenth_summary")
        if not tenth_summary:
            logger.error("No tenth_summary in state")
            return {
                "current_status": "save_tenth_summary_failed",
                "errors": [{"node": "save_tenth_summary", "error": "No summary to save"}],
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
                "current_status": "save_tenth_summary_failed",
                "errors": [{"node": "save_tenth_summary", "error": "No original record found"}],
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
            content=tenth_summary,
            compression_level=2,
            source_ids=[original_id],
            metadata={
                "type": "tenth_summary",
                "word_count": len(tenth_summary.split()),
                "chapter_count": len(state.get("chapters", [])),
            },
        )

        # Generate embedding and save to stores
        store_manager = get_store_manager()

        # Generate embedding
        embedding = await store_manager.embedding.embed(tenth_summary)
        record.embedding = embedding
        record.embedding_model = store_manager.embedding.model

        # Save to Elasticsearch
        await store_manager.es_stores.store.add(record)
        logger.info(f"Saved tenth summary record: {record_id}")

        # Create reference for state
        ref = StoreRecordRef(
            id=str(record_id),
            compression_level=2,
            content_preview=tenth_summary[:100],
        )

        return {
            "store_records": [ref],
            "current_status": "save_tenth_summary_complete",
        }

    except Exception as e:
        logger.error(f"Failed to save tenth summary: {e}")
        return {
            "current_status": "save_tenth_summary_failed",
            "errors": [{"node": "save_tenth_summary", "error": str(e)}],
        }
