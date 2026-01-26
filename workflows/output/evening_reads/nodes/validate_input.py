"""Input validation node for evening_reads workflow."""

import asyncio
import logging
import re
from typing import Any

from langchain_tools.base import get_store_manager

from ..state import CitationKeyMapping

logger = logging.getLogger(__name__)

CITATION_PATTERN = r"\[@([^\]]+)\]"
MIN_WORD_COUNT = 500


async def _lookup_es_record_for_key(
    store, zotero_key: str
) -> tuple[str | None, str | None]:
    """Look up ES record ID and title for a Zotero key.

    Returns (es_record_id, title) or (None, None) if not found.
    """
    query = {
        "bool": {
            "must": [{"term": {"zotero_key.keyword": zotero_key}}],
            "filter": [{"term": {"compression_level": 0}}],
        }
    }
    try:
        results = await store.search(query=query, size=1, compression_level=0)
        if results:
            record = results[0]
            title = record.metadata.get("title") if record.metadata else None
            return str(record.id), title
    except Exception as e:
        logger.warning(f"Failed to look up ES record for {zotero_key}: {e}")
    return None, None


async def validate_input_node(state: dict) -> dict[str, Any]:
    """Validate input and extract citation key mappings.

    Checks:
    - Literature review has minimum word count
    - Extracts all [@KEY] citations
    - Looks up ES record IDs for each citation key

    Returns:
        State update with is_valid, validation_error, extracted_citation_keys, citation_mappings
    """
    lit_review = state["input"]["literature_review"]

    # Check minimum length
    word_count = len(lit_review.split())
    if word_count < MIN_WORD_COUNT:
        logger.warning(f"Literature review too short: {word_count} words")
        return {
            "is_valid": False,
            "validation_error": (
                f"Literature review too short ({word_count} words). "
                f"Minimum {MIN_WORD_COUNT} words required."
            ),
            "extracted_citation_keys": [],
            "citation_mappings": {},
        }

    # Extract citation keys
    matches = re.findall(CITATION_PATTERN, lit_review)
    keys = set()
    for match in matches:
        # Handle multi-citations like [@key1; @key2]
        for key in match.split(";"):
            key = key.strip().lstrip("@")
            if key:
                keys.add(key)

    citation_keys = sorted(keys)

    if not citation_keys:
        logger.warning("No citations found in literature review")
        return {
            "is_valid": False,
            "validation_error": "No [@KEY] citations found in the literature review.",
            "extracted_citation_keys": [],
            "citation_mappings": {},
        }

    # Look up ES record IDs for each citation key
    store_manager = get_store_manager()
    store = store_manager.es_stores.store

    semaphore = asyncio.Semaphore(10)

    async def lookup_with_semaphore(key: str) -> tuple[str, str | None, str | None]:
        async with semaphore:
            es_id, title = await _lookup_es_record_for_key(store, key)
            return key, es_id, title

    results = await asyncio.gather(*[lookup_with_semaphore(k) for k in citation_keys])

    citation_mappings: dict[str, CitationKeyMapping] = {}
    found_count = 0
    for key, es_id, title in results:
        citation_mappings[key] = CitationKeyMapping(
            zotero_key=key,
            es_record_id=es_id,
            title=title,
        )
        if es_id:
            found_count += 1

    logger.info(
        f"Validated input: {word_count} words, {len(citation_keys)} unique citations, "
        f"{found_count} found in store"
    )

    return {
        "is_valid": True,
        "validation_error": None,
        "extracted_citation_keys": citation_keys,
        "citation_mappings": citation_mappings,
    }
