"""Test persistent caching functionality."""

import asyncio
import logging

from langchain_core.tracers.langchain import wait_for_all_tracers

from core.config import configure_logging
from langchain_tools.openalex.queries import get_work_by_doi
from workflows.shared.persistent_cache import (
    clear_cache,
    get_cache_stats,
    get_cached,
    set_cached,
)

configure_logging("cache_test")
logger = logging.getLogger(__name__)


async def test_openalex_cache():
    """Test OpenAlex metadata caching."""
    logger.info("=== Testing OpenAlex Cache ===")

    test_doi = "10.1038/s41586-020-2649-2"

    logger.info("First call (should fetch from API)...")
    work1 = await get_work_by_doi(test_doi)
    if work1:
        logger.info(f"Retrieved: {work1.title[:60]}...")
    else:
        logger.warning("DOI not found")
        return

    logger.info("Second call (should hit cache)...")
    work2 = await get_work_by_doi(test_doi)
    if work2:
        logger.info(f"Retrieved: {work2.title[:60]}...")

    logger.info(
        f"Results match: {work1.title == work2.title if work1 and work2 else False}"
    )


def test_manual_cache():
    """Test manual cache operations."""
    logger.info("=== Testing Manual Cache ===")

    cache_type = "test_cache"
    cache_key = "example_key"

    logger.info("Setting cache...")
    set_cached(cache_type, cache_key, {"data": "test_value", "number": 42})

    logger.info("Getting from cache...")
    result = get_cached(cache_type, cache_key, ttl_days=1)
    logger.info(f"Cached value: {result}")

    logger.info("Getting cache stats...")
    stats = get_cache_stats()
    logger.info(f"Cache stats: {stats}")

    logger.info("Clearing test cache...")
    clear_cache(cache_type)


async def test_cache_stats():
    """Display cache statistics."""
    logger.info("=== Cache Statistics ===")
    stats = get_cache_stats()

    for cache_type, info in stats.items():
        logger.info(f"{cache_type}: {info['files']} files, {info['size_mb']:.2f} MB")


async def main():
    """Run all cache tests."""
    test_manual_cache()
    await test_openalex_cache()
    await test_cache_stats()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Wait for LangSmith to flush all trace data before exiting
        wait_for_all_tracers()
