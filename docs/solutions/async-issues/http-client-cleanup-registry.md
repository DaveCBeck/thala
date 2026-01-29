---
module: System
date: 2026-01-10
problem_type: async_issue
component: async_task
symptoms:
  - "Unclosed client session warnings on shutdown"
  - "Unclosed aiohttp session warnings from Firecrawl"
  - "Resource leaks when tests terminate"
  - "Inconsistent cleanup patterns across HTTP client modules"
root_cause: connection_pool
resolution_type: code_fix
severity: medium
tags: [http-client, cleanup, resource-management, async, registry-pattern, lazy-initialization, httpx]
---

# HTTP Client Cleanup Registry

## Problem

Multiple async HTTP clients (Firecrawl, OpenAlex, BookSearch, Perplexity, Scraper) were initialized as global singletons with lazy initialization, but lacked centralized cleanup. This resulted in:

- "Unclosed client session" warnings in logs
- Resource leaks on application/test shutdown
- Each client managing its own cleanup independently (if at all)
- No single entry point for graceful shutdown

```
# Example warning
ResourceWarning: unclosed <httpx.AsyncClient object at 0x7f2b...>
sys:1: ResourceWarning: unclosed aiohttp session
```

## Root Cause

**Decentralized resource management**: Each service independently created and managed lazy-initialized async HTTP clients (`httpx.AsyncClient`, `AsyncFirecrawl`), with no unified shutdown mechanism.

Before the fix:
```python
# langchain_tools/book_search.py - NO cleanup registration
_book_search_client = None

def _get_client():
    global _book_search_client
    if _book_search_client is None:
        _book_search_client = httpx.AsyncClient(
            base_url=service_url,
            timeout=30.0,
        )
    return _book_search_client
# Client never closed on shutdown!
```

## Solution

**Add centralized cleanup registry with registration at lazy init time.**

### Core Registry (core/utils/async_http_client.py)

```python
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Global cleanup registry - list of (name, async_closer) tuples
_cleanup_registry: list[tuple[str, Callable[[], Awaitable[None]]]] = []


def register_cleanup(name: str, closer: Callable[[], Awaitable[None]]) -> None:
    """Register a cleanup function to be called on shutdown."""
    _cleanup_registry.append((name, closer))


async def cleanup_all_clients() -> None:
    """Close all registered HTTP clients (idempotent)."""
    for name, closer in _cleanup_registry:
        try:
            await closer()
        except Exception as e:
            logger.warning(f"Error closing {name} client: {e}")
```

### Client Registration Pattern

Each client defines a close function and registers during lazy initialization:

```python
# langchain_tools/book_search.py - WITH cleanup registration
_book_search_client = None


async def close_book_search() -> None:
    """Close the global book search client and release resources."""
    global _book_search_client
    if _book_search_client is not None:
        await _book_search_client.aclose()
        _book_search_client = None


def _get_client():
    """Get httpx AsyncClient for retrieve-academic service (lazy init)."""
    global _book_search_client
    if _book_search_client is None:
        import httpx
        from core.utils.async_http_client import register_cleanup

        service_url = os.environ.get("BOOK_SEARCH_SERVICE_URL", DEFAULT_SERVICE_URL)
        _book_search_client = httpx.AsyncClient(
            base_url=service_url,
            timeout=30.0,
        )
        register_cleanup("BookSearch", close_book_search)  # Register!
    return _book_search_client
```

### Usage at Shutdown

```python
async def main():
    try:
        # ... workflow execution ...
        result = await run_workflow()
    finally:
        # Clean up all HTTP clients
        from core.utils.async_http_client import cleanup_all_clients
        await cleanup_all_clients()

if __name__ == "__main__":
    asyncio.run(main())
```

## Registered Clients

| Client | Module | Close Function |
|--------|--------|----------------|
| BookSearch | `langchain_tools/book_search.py` | `close_book_search()` |
| Firecrawl | `langchain_tools/firecrawl.py` | `close_firecrawl()` |
| OpenAlex | `langchain_tools/openalex/client.py` | `close_openalex()` |
| Perplexity | `langchain_tools/perplexity.py` | `close_perplexity()` |
| Scraper | `core/scraping/service.py` | `close_scraper_service()` |

## Error Handling

The cleanup function continues even if individual clients fail:

```python
async def cleanup_all_clients() -> None:
    for name, closer in _cleanup_registry:
        try:
            await closer()
        except Exception as e:
            logger.warning(f"Error closing {name} client: {e}")
            # Continues to next client, doesn't propagate
```

## Files Modified

- `core/utils/async_http_client.py` - Added `register_cleanup()` and `cleanup_all_clients()`
- `core/utils/__init__.py` - Exported new functions
- `langchain_tools/book_search.py` - Added registration
- `langchain_tools/firecrawl.py` - Added registration
- `langchain_tools/openalex/client.py` - Added registration
- `langchain_tools/perplexity.py` - Added registration
- `core/scraping/service.py` - Added registration
- `testing/test_supervised_lit_review.py` - Added cleanup call

## Prevention

When adding new async HTTP clients:

1. **Define a close function** that checks for None and calls `.aclose()`
2. **Register during lazy init**: `register_cleanup("ClientName", close_function)`
3. **Call `cleanup_all_clients()`** in your test/workflow `finally` block

```python
# Template for new clients
_my_client = None

async def close_my_client() -> None:
    global _my_client
    if _my_client is not None:
        await _my_client.aclose()
        _my_client = None

def _get_my_client():
    global _my_client
    if _my_client is None:
        from core.utils.async_http_client import register_cleanup
        _my_client = httpx.AsyncClient(...)
        register_cleanup("MyClient", close_my_client)
    return _my_client
```

## Related Patterns

- [Streaming Producer-Consumer Pipeline](../../patterns/async-python/streaming-producer-consumer-pipeline.md) - Async task cleanup
- [Concurrent Scraping with TTL Cache](../../patterns/async-python/concurrent-scraping-with-ttl-cache.md) - Rate limiting and caching

## References

- [httpx AsyncClient](https://www.python-httpx.org/async/)
- [aiohttp ClientSession Cleanup](https://docs.aiohttp.org/en/stable/client_reference.html)
