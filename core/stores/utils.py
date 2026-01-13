"""Utility functions for store operations.

Cross-store verification and lookup utilities.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from .zotero import ZoteroStore
from .elasticsearch.client import ElasticsearchStores

logger = logging.getLogger(__name__)


@dataclass
class KeyVerificationResult:
    """Result of verifying a single Zotero key."""

    zotero_key: str
    exists_in_zotero: bool
    es_record_id: Optional[UUID] = None


async def verify_zotero_keys(
    keys: list[str],
    zotero_client: Optional[ZoteroStore] = None,
    es_stores: Optional[ElasticsearchStores] = None,
    concurrency: int = 10,
) -> list[KeyVerificationResult]:
    """Verify a list of Zotero keys and look up corresponding ES records.

    For each key:
    1. Checks if it exists in Zotero
    2. Looks up the corresponding ES record (L0) by zotero_key field

    Args:
        keys: List of 8-character Zotero citation keys
        zotero_client: Optional ZoteroStore instance (created if not provided)
        es_stores: Optional ElasticsearchStores instance (created if not provided)
        concurrency: Max concurrent verification requests

    Returns:
        List of KeyVerificationResult with:
        - zotero_key: The key that was verified
        - exists_in_zotero: True if item exists in Zotero library
        - es_record_id: UUID of the L0 ES record if found, None otherwise
    """
    if not keys:
        return []

    # Create clients if not provided
    own_zotero = zotero_client is None
    own_es = es_stores is None

    if own_zotero:
        zotero_client = ZoteroStore()
    if own_es:
        es_stores = ElasticsearchStores()

    try:
        semaphore = asyncio.Semaphore(concurrency)

        async def verify_single_key(key: str) -> KeyVerificationResult:
            """Verify a single key against Zotero and ES."""
            async with semaphore:
                # Check Zotero
                exists_in_zotero = await zotero_client.exists(key)

                # Look up ES record by zotero_key
                es_record_id = None
                try:
                    query = {"term": {"zotero_key": key}}
                    results = await es_stores.main.search(
                        query=query,
                        size=1,
                        compression_level=0,  # L0 only
                    )
                    if results:
                        es_record_id = results[0].id
                except Exception as e:
                    logger.warning(f"ES lookup failed for key '{key}': {e}")

                return KeyVerificationResult(
                    zotero_key=key,
                    exists_in_zotero=exists_in_zotero,
                    es_record_id=es_record_id,
                )

        # Run all verifications concurrently
        results = await asyncio.gather(*[verify_single_key(k) for k in keys])
        return list(results)

    finally:
        # Close clients we created
        if own_zotero:
            await zotero_client.close()
        if own_es:
            await es_stores.close()


async def verify_zotero_keys_batch(
    keys: set[str],
    zotero_client: Optional[ZoteroStore] = None,
    es_stores: Optional[ElasticsearchStores] = None,
) -> dict[str, KeyVerificationResult]:
    """Batch verify Zotero keys, returning a dict keyed by zotero_key.

    Convenience wrapper around verify_zotero_keys that returns
    results as a dict for easy lookup.

    Args:
        keys: Set of Zotero citation keys to verify
        zotero_client: Optional ZoteroStore instance
        es_stores: Optional ElasticsearchStores instance

    Returns:
        Dict mapping zotero_key -> KeyVerificationResult
    """
    results = await verify_zotero_keys(
        keys=list(keys),
        zotero_client=zotero_client,
        es_stores=es_stores,
    )
    return {r.zotero_key: r for r in results}
