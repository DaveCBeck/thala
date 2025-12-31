"""Document cache checking operations."""

import logging
from typing import Any, Optional

from langchain_tools.base import get_store_manager

logger = logging.getLogger(__name__)


async def check_document_exists_by_doi(doi: str) -> Optional[dict[str, Any]]:
    """Check if document already exists in ES L0 by DOI.

    Args:
        doi: The DOI to search for

    Returns:
        Dict with es_record_id, zotero_key, short_summary, content if found,
        None otherwise.
    """
    store_manager = get_store_manager()

    try:
        results = await store_manager.es_stores.store.search(
            query={
                "bool": {
                    "must": [
                        {"term": {"metadata.doi": doi}},
                        {"term": {"metadata.processing_status": "completed"}}
                    ]
                }
            },
            size=1,
            compression_level=0,
        )

        if results:
            record = results[0]
            return {
                "es_record_id": str(record.id),
                "zotero_key": record.zotero_key,
                "content": record.content,
                "short_summary": record.metadata.get("short_summary", ""),
            }
    except Exception as e:
        logger.debug(f"ES lookup for DOI {doi} failed: {e}")

    return None
