"""
Save 10:1 summary as StoreRecord node.

Creates dual L2 records for non-English documents:
- L2 record with original language summary (language_code=original)
- L2-en record with English translation (language_code="en")
"""

import logging
from typing import Any
from uuid import UUID, uuid4

from langsmith import traceable

from core.stores.schema import SourceType, StoreRecord
from langchain_tools.base import get_store_manager
from workflows.document_processing.state import DocumentProcessingState, StoreRecordRef

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="SaveTenthSummary")
async def save_tenth_summary(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Create StoreRecord(s) for the 10:1 summary.

    For English documents: creates one L2 record.
    For non-English: creates two L2 records (original + English translation).

    Returns updated store_records list and current_status.
    """
    try:
        # Prefer new field, fall back to old for compatibility
        tenth_summary_original = state.get("tenth_summary_original") or state.get(
            "tenth_summary"
        )
        tenth_summary_english = state.get("tenth_summary_english")
        original_language = state.get("original_language", "en")

        if not tenth_summary_original:
            logger.error("No tenth_summary in state")
            return {
                "current_status": "save_tenth_summary_failed",
                "errors": [
                    {"node": "save_tenth_summary", "error": "No summary to save"}
                ],
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
                "errors": [
                    {"node": "save_tenth_summary", "error": "No original record found"}
                ],
            }

        store_manager = get_store_manager()
        zotero_key = state.get("zotero_key")
        chapter_count = len(state.get("chapters", []))

        # Convert original record ID to UUID
        original_id = original_record["id"]
        if isinstance(original_id, str):
            original_id = UUID(original_id)

        new_refs = []

        # Create original language L2 record
        record_id_original = uuid4()
        record_original = StoreRecord(
            id=record_id_original,
            source_type=SourceType.INTERNAL,
            zotero_key=zotero_key,
            content=tenth_summary_original,
            compression_level=2,
            source_ids=[original_id],
            language_code=original_language,
            metadata={
                "type": "tenth_summary",
                "word_count": len(tenth_summary_original.split()),
                "chapter_count": chapter_count,
                "is_original_language": True,
            },
        )

        # Generate embedding (use embed_long to handle large summaries)
        embedding = await store_manager.embedding.embed_long(tenth_summary_original)
        record_original.embedding = embedding
        record_original.embedding_model = store_manager.embedding.model
        await store_manager.es_stores.store.add(record_original)

        logger.info(
            f"Saved original language L2 record: {record_id_original} (lang={original_language})"
        )

        new_refs.append(
            StoreRecordRef(
                id=str(record_id_original),
                compression_level=2,
                content_preview=tenth_summary_original[:100],
                language_code=original_language,
            )
        )

        # Create English translation L2 record if non-English
        if original_language != "en" and tenth_summary_english:
            record_id_english = uuid4()
            record_english = StoreRecord(
                id=record_id_english,
                source_type=SourceType.INTERNAL,
                zotero_key=zotero_key,
                content=tenth_summary_english,
                compression_level=2,
                source_ids=[original_id],  # Same source - the L0 record
                language_code="en",
                metadata={
                    "type": "tenth_summary",
                    "word_count": len(tenth_summary_english.split()),
                    "chapter_count": chapter_count,
                    "is_original_language": False,
                    "translated_from": original_language,
                },
            )

            embedding_en = await store_manager.embedding.embed_long(
                tenth_summary_english
            )
            record_english.embedding = embedding_en
            record_english.embedding_model = store_manager.embedding.model
            await store_manager.es_stores.store.add(record_english)

            logger.info(f"Saved English translation L2 record: {record_id_english}")

            new_refs.append(
                StoreRecordRef(
                    id=str(record_id_english),
                    compression_level=2,
                    content_preview=tenth_summary_english[:100],
                    language_code="en",
                )
            )

        return {
            "store_records": new_refs,
            "current_status": "save_tenth_summary_complete",
        }

    except Exception as e:
        logger.error(f"Failed to save tenth summary: {e}")
        return {
            "current_status": "save_tenth_summary_failed",
            "errors": [{"node": "save_tenth_summary", "error": str(e)}],
        }
