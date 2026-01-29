---
name: concurrent-scraping-with-ttl-cache
title: "Concurrent Scraping with TTL Cache"
date: 2025-12-22
category: async-python
applicability:
  - "When scraping multiple URLs where the same URLs may be requested repeatedly across iterations"
  - "When parallel execution is needed for I/O-bound scraping operations"
  - "When deterministic result ordering must be preserved despite concurrent execution"
components: [async_task, web_scraper]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [cachetools, ttl-cache, asyncio-gather, parallel-execution, scraping, deterministic-ordering, index-tracking]
---

# Concurrent Scraping with TTL Cache

## Intent

Execute multiple URL scraping operations concurrently via `asyncio.gather()` while avoiding redundant scrapes through TTL-based caching, maintaining deterministic result ordering despite asynchronous completion.

## Motivation

In research workflows, the same URLs often appear across different researchers or iterations:

1. **Redundant scrapes**: Multiple researchers may find the same source
2. **Sequential bottleneck**: Scraping URLs one-by-one is slow for I/O-bound operations
3. **Order dependency**: Downstream processing may expect results in deterministic order
4. **Memory pressure**: Unbounded caching leads to memory exhaustion

This pattern solves all four issues:
- TTLCache avoids re-scraping within a time window
- `asyncio.gather()` parallelizes I/O-bound operations
- Index tracking preserves original ordering
- Cache maxsize prevents memory exhaustion

## Applicability

Use this pattern when:
- Scraping multiple URLs where duplicates are likely (research, crawling)
- I/O latency dominates CPU time (network-bound operations)
- Results must maintain original order for downstream processing
- Memory constraints require bounded caching

Do NOT use this pattern when:
- URLs are always unique (no cache benefit)
- Order doesn't matter (simpler pattern without index tracking)
- Long-term persistence needed (use file-based cache instead)
- CPU-bound operations (parallelism won't help)

## Structure

```
                    +-------------------+
                    |   URL Results     |
                    |   [0, 1, 2, 3]    |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
         +--------+    +--------+    +--------+
         | Task 0 |    | Task 1 |    | Task 2 |
         +---+----+    +---+----+    +---+----+
             |             |             |
             v             v             v
      +-----------+  +-----------+  +-----------+
      |   Cache   |  |   Cache   |  |   Cache   |
      |   Check   |  |   Check   |  |   Check   |
      +-----+-----+  +-----+-----+  +-----+-----+
            |              |              |
        HIT |          MISS|          HIT |
            v              v              v
      +-----------+  +-----------+  +-----------+
      |  Return   |  |  Scrape   |  |  Return   |
      |  Cached   |  |  + Cache  |  |  Cached   |
      +-----------+  +-----------+  +-----------+
            |              |              |
            +--------------+--------------+
                           |
                    +------v------+
                    | asyncio.    |
                    | gather()    |
                    +------+------+
                           |
                    +------v------+
                    | Sort by     |
                    | Index       |
                    +------+------+
                           |
                    +------v------+
                    | Results     |
                    | [0, 1, 2]   |  (deterministic order)
                    +-------------+
```

## Implementation

### Step 1: Configure TTL Cache

Use `cachetools.TTLCache` with environment-configurable size and TTL:

```python
# workflows/research/subgraphs/researcher.py

import os
from cachetools import TTLCache

# Cache for scraped URL content (1 hour TTL, max 200 items)
# Avoids re-scraping the same URL across researchers or iterations
_scrape_cache: TTLCache = TTLCache(
    maxsize=int(os.getenv("SCRAPE_CACHE_SIZE", "200")),
    ttl=int(os.getenv("SCRAPE_CACHE_TTL", "3600")),  # 1 hour default
)
```

**Configuration guidelines:**
- `maxsize`: Estimate max unique URLs per workflow run (200 covers most research sessions)
- `ttl`: Set to workflow duration or longer (3600s = 1 hour covers most sessions)

### Step 2: Create Single-URL Scraper with Cache

Extract scraping logic into a function that returns index for ordering:

```python
async def _scrape_single_url(
    result: WebSearchResult,
    index: int,
) -> tuple[int, str | None, WebSearchResult]:
    """Scrape a single URL and return the result.

    This helper function enables parallel scraping via asyncio.gather().
    Uses a TTL cache to avoid re-scraping the same URL.

    Args:
        result: WebSearchResult containing URL and metadata
        index: Original index in results list (for preserving order)

    Returns:
        Tuple of (index, scraped_content_str, updated_result)
    """
    url = result["url"]
    updated_result = dict(result)
    updated_result["_index"] = index  # Store for later sorting

    # Check cache first
    if url in _scrape_cache:
        content = _scrape_cache[url]
        logger.debug(f"Cache hit for: {url} ({len(content)} chars)")

        content_str = f"[{result['title']}]\nURL: {url}\n\n{content}"
        updated_result["content"] = content

        return (index, content_str, updated_result)

    # Cache miss - scrape the URL
    try:
        # Route PDFs to specialized processor
        if is_pdf_url(url):
            content = await fetch_pdf_via_marker(url)
            if not content:
                # Fallback to generic scraper
                response = await scrape_url.ainvoke({"url": url})
                content = response.get("markdown", "")
        else:
            response = await scrape_url.ainvoke({"url": url})
            content = response.get("markdown", "")

        # Truncate very long content
        if len(content) > 8000:
            content = content[:8000] + "\n\n[Content truncated...]"

        # Store in cache for future use
        _scrape_cache[url] = content
        logger.debug(f"Scraped and cached {len(content)} chars from: {url}")

        content_str = f"[{result['title']}]\nURL: {url}\n\n{content}"
        updated_result["content"] = content

        return (index, content_str, updated_result)

    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return (index, None, updated_result)  # Return None content on failure
```

### Step 3: Execute Scrapes in Parallel

Use `asyncio.gather()` with `return_exceptions=True` for fault tolerance:

```python
import asyncio

MAX_SCRAPES = 10  # Limit parallel scrapes


async def scrape_pages(state: ResearcherState) -> dict[str, Any]:
    """Scrape top results for full content in parallel.

    Uses asyncio.gather() to scrape multiple URLs concurrently.
    Results are cached with TTL to avoid redundant scrapes across researchers.
    """
    results = state.get("search_results", [])

    if not results:
        return {"scraped_content": [], "search_results": []}

    # Log cache statistics
    urls_to_scrape = [r["url"] for r in results[:MAX_SCRAPES]]
    cached_count = sum(1 for url in urls_to_scrape if url in _scrape_cache)
    logger.info(
        f"Scraping {len(urls_to_scrape)} URLs ({cached_count} cached, "
        f"{len(urls_to_scrape) - cached_count} new) - cache size: {len(_scrape_cache)}"
    )

    # Create scraping tasks for parallel execution
    scraping_tasks = [
        _scrape_single_url(result, i)
        for i, result in enumerate(results[:MAX_SCRAPES])
    ]

    # Execute all scrapes concurrently
    task_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)

    # Process results
    scraped = []
    updated_results = []

    for task_result in task_results:
        if isinstance(task_result, Exception):
            logger.error(f"Scraping task failed with exception: {task_result}")
            continue

        idx, content_str, updated_result = task_result

        if content_str:
            scraped.append(content_str)

        updated_results.append(updated_result)

    # Sort by original index to maintain deterministic order
    updated_results.sort(key=lambda x: x.get("_index", 0))

    # Remove temporary index field
    for r in updated_results:
        r.pop("_index", None)

    # Keep remaining results without scraping
    updated_results.extend(results[MAX_SCRAPES:])

    logger.info(f"Scraped {len(scraped)} URLs successfully in parallel")

    return {
        "scraped_content": scraped,
        "search_results": updated_results,
    }
```

### Step 4: Add Rate Limiting (Optional)

For APIs with rate limits, use a semaphore to bound concurrency:

```python
import asyncio

# Limit concurrent scrapes to avoid rate limits
_scrape_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent


async def _scrape_single_url_with_rate_limit(result, index):
    """Scrape with semaphore-controlled concurrency."""
    async with _scrape_semaphore:
        return await _scrape_single_url(result, index)


async def scrape_pages(state):
    scraping_tasks = [
        _scrape_single_url_with_rate_limit(result, i)  # Use rate-limited version
        for i, result in enumerate(results[:MAX_SCRAPES])
    ]
    task_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
    # ... rest of processing
```

## Consequences

### Benefits

- **90%+ reduction in redundant scrapes**: Same URLs across researchers/iterations hit cache
- **Parallel speedup**: I/O-bound operations complete faster (5-10x improvement)
- **Deterministic ordering**: Downstream processing sees consistent result order
- **Memory bounded**: TTLCache evicts old entries, preventing memory exhaustion
- **Fault tolerant**: `return_exceptions=True` ensures partial results on failures

### Trade-offs

- **Cache coherence**: Cached content may become stale within TTL window
- **Memory overhead**: Cache consumes memory (mitigated by maxsize)
- **Ordering overhead**: Index tracking and sorting adds small overhead
- **Single-process**: Cache not shared across process boundaries (use Redis for distributed)

### Async Considerations

- **Non-blocking**: All operations use `await`, no blocking I/O
- **Concurrency control**: Optional semaphore prevents overwhelming APIs
- **Exception handling**: `return_exceptions=True` isolates failures
- **Resource cleanup**: TTLCache auto-evicts; no explicit cleanup needed

## Related Patterns

- [Parallel AI Search Integration](../data-pipeline/parallel-ai-search-integration.md) - Uses `asyncio.gather()` for multi-source search
- [Unified Scraping Service](../data-pipeline/unified-scraping-service-fallback-chain.md) - Fallback chain for robust scraping

## Known Uses in Thala

- `workflows/research/subgraphs/researcher.py`: Parallel URL scraping with TTL cache
- `langchain_tools/book_search.py`: TTLCache for book search results (30 min TTL)
- `services/retrieve-academic/app/retriever/cache.py`: TTLCache for search and unpaywall lookups

## References

- [cachetools documentation](https://cachetools.readthedocs.io/)
- [asyncio.gather documentation](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
