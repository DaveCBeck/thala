"""HTTP client management for OpenAlex."""

import os

_openalex_client = None


async def close_openalex() -> None:
    """Close the global OpenAlex client and release resources."""
    global _openalex_client
    if _openalex_client is not None:
        await _openalex_client.aclose()
        _openalex_client = None


def _get_openalex():
    """Get OpenAlex httpx client (lazy init)."""
    global _openalex_client
    if _openalex_client is None:
        import httpx
        from core.utils.async_http_client import register_cleanup

        # OpenAlex recommends providing email for polite pool (faster rate limits)
        email = os.environ.get("OPENALEX_EMAIL", "")
        headers = {}
        if email:
            headers["User-Agent"] = f"mailto:{email}"

        _openalex_client = httpx.AsyncClient(
            base_url="https://api.openalex.org",
            headers=headers,
            timeout=30.0,
        )
        register_cleanup("OpenAlex", close_openalex)
    return _openalex_client
