"""
Update the initial store record with processed content.

Documents go to Elasticsearch `store` only. Chroma (top_of_mind) is reserved
for user-curated active context, not automatically populated.
"""

import logging
from uuid import UUID

from langchain_tools.base import get_store_manager

logger = logging.getLogger(__name__)


async def update_store(state: dict) -> dict:
    """
    Update the initial store record with processed content.

    Updates the StoreRecord in Elasticsearch with:
    - Full markdown content
    - Processing metadata (word_count, page_count, ocr_method)

    Note: Documents are NOT added to Chroma. The `top_of_mind` store (Chroma)
    is reserved for user-curated active context only.
    """
    store_manager = get_store_manager()
    processing_result = state["processing_result"]
    store_records = state["store_records"]

    if not store_records:
        raise ValueError("No store records found to update")

    # Get the initial record ID
    record_id = UUID(store_records[0]["id"])

    # Update Elasticsearch record with full content and metadata
    # Original documents are stored in store_l0 (compression_level=0)
    await store_manager.es_stores.store.update(
        record_id,
        {
            "content": processing_result["markdown"],
            "metadata": {
                "word_count": processing_result["word_count"],
                "page_count": processing_result["page_count"],
                "ocr_method": processing_result.get("ocr_method", "none"),
                "chunk_count": len(processing_result.get("chunks", [])),
                "processing_status": "completed",
            },
        },
        compression_level=0,
    )

    logger.info(
        f"Updated store record {record_id}: "
        f"{processing_result['word_count']} words, "
        f"{processing_result['page_count']} pages, "
        f"{len(processing_result.get('chunks', []))} chunks"
    )

    # Return updated store records info
    updated_store_records = [{
        "id": str(record_id),
        "compression_level": 0,
        "content_preview": processing_result["markdown"][:200],
    }]

    return {
        "store_records": updated_store_records,
        "current_status": "store_updated",
    }
