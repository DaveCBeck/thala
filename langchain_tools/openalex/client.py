"""HTTP client management for OpenAlex."""

import os

_openalex_client = None


class _SemaphoreClient:
    """Thin wrapper around httpx.AsyncClient that applies a semaphore to .get() calls."""

    def __init__(self, client):
        self._client = client

    async def get(self, *args, **kwargs):
        from core.task_queue.rate_limits import get_openalex_semaphore

        async with get_openalex_semaphore():
            return await self._client.get(*args, **kwargs)

    async def aclose(self):
        await self._client.aclose()


async def close_openalex() -> None:
    """Close the global OpenAlex client and release resources."""
    global _openalex_client
    if _openalex_client is not None:
        await _openalex_client.aclose()
        _openalex_client = None


def _get_openalex():
    """Get OpenAlex httpx client (lazy init).

    Returns a thin wrapper that applies a global semaphore to .get() calls,
    limiting concurrent OpenAlex requests during parallel workflow execution.
    """
    global _openalex_client
    if _openalex_client is None:
        import httpx
        from core.utils.async_http_client import register_cleanup

        # OpenAlex recommends providing email for polite pool (faster rate limits)
        email = os.environ.get("OPENALEX_EMAIL", "")
        headers = {}
        if email:
            headers["User-Agent"] = f"mailto:{email}"

        raw_client = httpx.AsyncClient(
            base_url="https://api.openalex.org",
            headers=headers,
            timeout=30.0,
        )
        _openalex_client = _SemaphoreClient(raw_client)
        register_cleanup("OpenAlex", close_openalex)
    return _openalex_client
