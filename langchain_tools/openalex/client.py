"""HTTP client management for OpenAlex."""

import os

_openalex_client = None


def _get_openalex():
    """Get OpenAlex httpx client (lazy init)."""
    global _openalex_client
    if _openalex_client is None:
        import httpx

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
    return _openalex_client
