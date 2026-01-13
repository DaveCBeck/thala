"""Utility functions for citation processing."""

from urllib.parse import urlparse


def _normalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    parsed = urlparse(url)
    # Remove trailing slashes, normalize to lowercase domain
    path = parsed.path.rstrip("/") if parsed.path != "/" else ""
    return f"{parsed.scheme}://{parsed.netloc.lower()}{path}"
