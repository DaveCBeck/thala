"""Zotero-specific citation validation."""

import asyncio
import logging
from typing import Set, Optional

from core.stores.zotero import ZoteroStore

from .parsers import extract_citation_keys_from_text, generate_fallback_key

logger = logging.getLogger(__name__)


async def verify_zotero_citation(key: str, zotero_client: ZoteroStore) -> bool:
    """Verify that a Zotero citation key exists."""
    try:
        return await zotero_client.exists(key)
    except Exception as e:
        logger.warning(f"Zotero verification failed for key '{key}': {e}")
        return False


async def verify_zotero_citations_batch(
    keys: Set[str],
    zotero_client: ZoteroStore,
    concurrency: int = 10,
) -> dict[str, bool]:
    """Verify multiple Zotero citation keys in batch."""
    if not keys:
        return {}

    semaphore = asyncio.Semaphore(concurrency)

    async def verify_with_semaphore(key: str) -> tuple[str, bool]:
        async with semaphore:
            exists = await verify_zotero_citation(key, zotero_client)
            return key, exists

    results = await asyncio.gather(*[verify_with_semaphore(k) for k in keys])
    return dict(results)


async def validate_citations_against_zotero(
    text: str,
    zotero_client: ZoteroStore,
    known_valid_keys: Optional[Set[str]] = None,
) -> tuple[Set[str], Set[str]]:
    """Validate all citations in text against Zotero."""
    all_keys = extract_citation_keys_from_text(text)

    if not all_keys:
        return set(), set()

    known_valid = known_valid_keys or set()
    keys_to_verify = all_keys - known_valid

    if not keys_to_verify:
        return all_keys, set()

    verification_results = await verify_zotero_citations_batch(keys_to_verify, zotero_client)

    valid_keys = known_valid.copy()
    invalid_keys = set()

    for key, exists in verification_results.items():
        if exists:
            valid_keys.add(key)
        else:
            invalid_keys.add(key)

    return valid_keys, invalid_keys


def validate_corpus_zotero_keys(
    paper_summaries: dict,
    zotero_keys: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Validate that all corpus papers have Zotero keys and generate missing ones."""
    missing_keys: list[str] = []
    updated_keys = dict(zotero_keys)

    for doi in paper_summaries.keys():
        if doi not in zotero_keys:
            fallback_key = generate_fallback_key(doi, paper_summaries.get(doi, {}))
            updated_keys[doi] = fallback_key
            missing_keys.append(doi)
            logger.warning(
                f"Generated fallback Zotero key '{fallback_key}' for paper {doi}"
            )

    if missing_keys:
        logger.info(
            f"Generated {len(missing_keys)} fallback Zotero keys for corpus papers"
        )

    return missing_keys, updated_keys
