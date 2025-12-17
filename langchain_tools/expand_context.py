"""
expand_context - Deep-dive retrieval tool for LangChain.

Fetches more detail on a specific record or source ("more about that").
Supports UUID, zotero_key, or content snippet (fuzzy match).
"""

import logging
import re
from typing import Literal, Optional
from uuid import UUID

from langchain.tools import tool
from pydantic import BaseModel, Field

from .base import get_store_manager

logger = logging.getLogger(__name__)

# UUID regex pattern
UUID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE
)

# Zotero key pattern: exactly 8 alphanumeric characters
ZOTERO_KEY_PATTERN = re.compile(r'^[A-Za-z0-9]{8}$')


class ExpandedContext(BaseModel):
    """Output schema for expand_context tool."""

    reference: str
    reference_type: str
    found: bool

    # Store record data (if UUID or content match)
    store_record: Optional[dict] = None
    source_store: Optional[str] = None

    # Zotero data (if zotero_key present or direct lookup)
    zotero_item: Optional[dict] = None

    # History (if requested)
    history: Optional[list[dict]] = None
    history_count: int = 0

    # For content matches, include match confidence
    match_info: Optional[str] = None


def _detect_reference_type(reference: str, hint: str) -> str:
    """Auto-detect reference type based on format."""
    if hint != "auto":
        return hint

    # Check for UUID format
    if UUID_PATTERN.match(reference):
        return "uuid"

    # Check for Zotero key (exactly 8 alphanumeric)
    if ZOTERO_KEY_PATTERN.match(reference):
        return "zotero_key"

    # Default to content snippet
    return "content"


async def _expand_by_uuid(
    reference: str,
    result: ExpandedContext,
    include_zotero: bool,
    include_history: bool,
) -> None:
    """Look up record by UUID across all stores."""
    store_manager = get_store_manager()

    try:
        record_id = UUID(reference)
    except ValueError:
        result.match_info = "Invalid UUID format"
        return

    record = None
    source_store = None

    # Try each store in order of likelihood
    # 1. Check store (most common)
    record = await store_manager.es_stores.store.get(record_id)
    if record:
        source_store = "store"

    # 2. Check coherence
    if not record:
        record = await store_manager.es_stores.coherence.get(record_id)
        if record:
            source_store = "coherence"

    # 3. Check top_of_mind (ChromaDB)
    if not record:
        chroma_result = await store_manager.chroma.get(record_id)
        if chroma_result:
            record = chroma_result
            source_store = "top_of_mind"

    if record:
        result.found = True
        result.source_store = source_store

        # Convert to dict
        if hasattr(record, "model_dump"):
            result.store_record = record.model_dump(mode="json")
        else:
            result.store_record = record  # Chroma returns dict

        # Get zotero_key from record
        zotero_key = None
        if isinstance(record, dict):
            zotero_key = record.get("metadata", {}).get("zotero_key")
            if not zotero_key:
                zotero_key = record.get("zotero_key")
        elif hasattr(record, "zotero_key"):
            zotero_key = record.zotero_key

        # Fetch linked Zotero if present
        if include_zotero and zotero_key:
            await _fetch_zotero(zotero_key, result)

        # Fetch history if requested
        if include_history:
            await _fetch_history(record_id, result)


async def _expand_by_zotero_key(
    zotero_key: str,
    result: ExpandedContext,
) -> None:
    """Direct Zotero lookup by key."""
    store_manager = get_store_manager()

    try:
        zotero_item = await store_manager.zotero.get(zotero_key)
        if zotero_item:
            result.found = True
            result.zotero_item = zotero_item.model_dump(mode="json")
    except Exception as e:
        logger.warning(f"Zotero lookup failed for {zotero_key}: {e}")
        result.match_info = f"Zotero lookup failed: {e}"


async def _expand_by_content(
    content_snippet: str,
    result: ExpandedContext,
    include_zotero: bool,
    include_history: bool,
) -> None:
    """Fuzzy match by content snippet - search all stores and return best match."""
    store_manager = get_store_manager()

    best_match = None
    best_source = None
    best_score = -1.0

    # 1. Semantic search in top_of_mind (most likely to find fuzzy matches)
    try:
        query_embedding = await store_manager.embedding.embed(content_snippet)
        chroma_results = await store_manager.chroma.search(
            query_embedding=query_embedding,
            n_results=1,
        )
        if chroma_results:
            match = chroma_results[0]
            similarity = 1 - match["distance"]
            if similarity > best_score:
                best_match = match
                best_source = "top_of_mind"
                best_score = similarity
    except Exception as e:
        logger.warning(f"top_of_mind search failed: {e}")

    # 2. Text search in coherence
    try:
        coherence_results = await store_manager.es_stores.coherence.search(
            query={"match": {"content": content_snippet}},
            size=1,
        )
        if coherence_results:
            # We don't have comparable scores, but if semantic search found nothing,
            # take this as a reasonable match
            if best_score < 0.5:  # Only if semantic match is weak
                best_match = coherence_results[0]
                best_source = "coherence"
    except Exception as e:
        logger.warning(f"coherence search failed: {e}")

    # 3. Text search in store
    try:
        store_results = await store_manager.es_stores.store.search(
            query={"match": {"content": content_snippet}},
            size=1,
        )
        if store_results and best_match is None:
            best_match = store_results[0]
            best_source = "store"
    except Exception as e:
        logger.warning(f"store search failed: {e}")

    if best_match:
        result.found = True
        result.source_store = best_source
        result.match_info = f"Fuzzy match from {best_source}"

        if best_score > 0:
            result.match_info += f" (similarity: {best_score:.2f})"

        # Convert to dict
        if hasattr(best_match, "model_dump"):
            result.store_record = best_match.model_dump(mode="json")
        else:
            result.store_record = best_match

        # Get record ID for history lookup
        record_id = None
        if isinstance(best_match, dict):
            record_id = best_match.get("id")
        elif hasattr(best_match, "id"):
            record_id = best_match.id

        # Get zotero_key
        zotero_key = None
        if isinstance(best_match, dict):
            zotero_key = best_match.get("metadata", {}).get("zotero_key")
            if not zotero_key:
                zotero_key = best_match.get("zotero_key")
        elif hasattr(best_match, "zotero_key"):
            zotero_key = best_match.zotero_key

        if include_zotero and zotero_key:
            await _fetch_zotero(zotero_key, result)

        if include_history and record_id:
            if isinstance(record_id, str):
                record_id = UUID(record_id)
            await _fetch_history(record_id, result)
    else:
        result.match_info = "No matching records found"


async def _fetch_zotero(zotero_key: str, result: ExpandedContext) -> None:
    """Fetch Zotero item and add to result."""
    store_manager = get_store_manager()

    try:
        zotero_item = await store_manager.zotero.get(zotero_key)
        if zotero_item:
            result.zotero_item = zotero_item.model_dump(mode="json")
    except Exception as e:
        logger.warning(f"Zotero fetch failed for {zotero_key}: {e}")


async def _fetch_history(record_id: UUID, result: ExpandedContext) -> None:
    """Fetch edit history from who_i_was."""
    store_manager = get_store_manager()

    try:
        history = await store_manager.es_stores.who_i_was.get_history(record_id)
        result.history = [h.model_dump(mode="json") for h in history]
        result.history_count = len(history)
    except Exception as e:
        logger.warning(f"History fetch failed for {record_id}: {e}")


@tool
async def expand_context(
    reference: str,
    reference_type: Literal["uuid", "zotero_key", "content", "auto"] = "auto",
    include_zotero: bool = True,
    include_history: bool = False,
) -> dict:
    """Get more detail about a specific memory record or source.

    Use this when you:
    - Have a record ID and need the full content
    - Want to see the source material (Zotero) for a record
    - Need to understand how a belief evolved over time
    - Remember something vaguely and want to find the full record

    Accepts UUID, 8-character Zotero key, or content snippet for fuzzy match.

    Args:
        reference: UUID of a store record, 8-character Zotero key, or content snippet
        reference_type: Type of reference. 'auto' detects based on format.
        include_zotero: If record has zotero_key, also fetch Zotero metadata.
        include_history: Include edit history from who_i_was.
    """
    # Auto-detect reference type
    detected_type = _detect_reference_type(reference, reference_type)

    result = ExpandedContext(
        reference=reference,
        reference_type=detected_type,
        found=False,
    )

    if detected_type == "uuid":
        await _expand_by_uuid(
            reference, result, include_zotero, include_history
        )
    elif detected_type == "zotero_key":
        await _expand_by_zotero_key(reference, result)
    elif detected_type == "content":
        await _expand_by_content(
            reference, result, include_zotero, include_history
        )

    return result.model_dump(mode="json")
