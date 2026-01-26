"""Content fetching node for evening_reads workflow.

Fetches L2 (10:1 summary) or L0 (original) content from the store for each
anchor paper in a deep-dive assignment.
"""

import logging
from typing import Any, Literal
from uuid import UUID

from langchain_tools.base import get_store_manager

from ..state import EnrichedContent

logger = logging.getLogger(__name__)

# Threshold for auto-generating L2 from large L0 documents
L0_SIZE_THRESHOLD_FOR_L2 = 150_000


async def _fetch_content_for_key(
    store, zotero_key: str, es_record_id: str | None
) -> tuple[str | None, Literal["L0", "L2"] | None]:
    """Fetch content for a single Zotero key, preferring L2 over L0.

    This mirrors the logic from paper_processor/extraction/parsers.py.

    Args:
        store: MainStore instance
        zotero_key: Zotero citation key
        es_record_id: Pre-looked-up ES record ID (L0), or None to search

    Returns:
        (content, level) tuple, or (None, None) if not found
    """
    # If we don't have the ES record ID, look it up
    if es_record_id is None:
        query = {
            "bool": {
                "must": [{"term": {"zotero_key.keyword": zotero_key}}],
                "filter": [{"term": {"compression_level": 0}}],
            }
        }
        try:
            results = await store.search(query=query, size=1, compression_level=0)
            if results:
                es_record_id = str(results[0].id)
            else:
                logger.warning(f"No L0 record found for zotero_key: {zotero_key}")
                return None, None
        except Exception as e:
            logger.error(f"Failed to search for L0 record: {e}")
            return None, None

    record_uuid = UUID(es_record_id)

    # Try L2 first (10:1 summary) - better for long documents
    try:
        l2_record = await store.get_by_source_id(record_uuid, compression_level=2)
        if l2_record and l2_record.content:
            logger.debug(f"Using L2 (10:1 summary) for {zotero_key}")
            return l2_record.content, "L2"
    except Exception as e:
        logger.warning(f"Failed to fetch L2 for {zotero_key}: {e}")

    # Fall back to L0 (original)
    try:
        l0_record = await store.get(record_uuid, compression_level=0)
        if l0_record and l0_record.content:
            # Check if L0 is too large
            if len(l0_record.content) > L0_SIZE_THRESHOLD_FOR_L2:
                logger.warning(
                    f"L0 for {zotero_key} is {len(l0_record.content)} chars "
                    f"(>{L0_SIZE_THRESHOLD_FOR_L2}), may need truncation"
                )
            logger.debug(f"Using L0 (original) for {zotero_key}")
            return l0_record.content, "L0"
    except Exception as e:
        logger.error(f"Failed to fetch L0 for {zotero_key}: {e}")

    return None, None


async def fetch_content_node(state: dict) -> dict[str, Any]:
    """Fetch content for a single deep-dive assignment.

    This node is called via Send() with the assignment details.
    It fetches content for all anchor papers in the assignment.

    Expected state keys from Send():
        - deep_dive_id: Which deep-dive this is for
        - anchor_keys: List of Zotero keys to fetch
        - citation_mappings: Dict of zotero_key -> CitationKeyMapping

    Returns:
        State update with enriched_content list (aggregated via add reducer)
    """
    deep_dive_id = state.get("deep_dive_id")
    anchor_keys = state.get("anchor_keys", [])
    citation_mappings = state.get("citation_mappings", {})

    if not deep_dive_id or not anchor_keys:
        logger.error(f"Missing required state: deep_dive_id={deep_dive_id}, anchor_keys={anchor_keys}")
        return {
            "errors": [{"node": "fetch_content", "error": "Missing deep_dive_id or anchor_keys"}]
        }

    store_manager = get_store_manager()
    store = store_manager.es_stores.store

    enriched: list[EnrichedContent] = []

    for key in anchor_keys:
        mapping = citation_mappings.get(key, {})
        es_record_id = mapping.get("es_record_id") if mapping else None

        content, level = await _fetch_content_for_key(store, key, es_record_id)

        if content and level:
            enriched.append(
                EnrichedContent(
                    deep_dive_id=deep_dive_id,
                    zotero_key=key,
                    content=content,
                    content_level=level,
                )
            )
            logger.info(f"Fetched {level} content for {key} ({len(content)} chars)")
        else:
            logger.warning(f"No content found for {key}")

    if not enriched:
        logger.warning(f"No content fetched for deep-dive {deep_dive_id}")

    return {"enriched_content": enriched}
