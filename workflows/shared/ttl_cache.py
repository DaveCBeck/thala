"""TTL cache utilities for workflow data."""

import os
from typing import Any

from cachetools import TTLCache

# Registry of named caches (module-level singletons)
_caches: dict[str, TTLCache] = {}


def get_ttl_cache(
    name: str,
    *,
    maxsize: int | None = None,
    ttl: int | None = None,
    env_prefix: str | None = None,
) -> TTLCache:
    """Get or create a named TTL cache.

    Caches are singletons - calling with the same name returns the same instance.
    Configuration can come from env vars or explicit parameters.

    Args:
        name: Unique cache name (used for registry and env var prefix)
        maxsize: Maximum items (default: 200, or from {ENV_PREFIX}_CACHE_SIZE)
        ttl: TTL in seconds (default: 3600, or from {ENV_PREFIX}_CACHE_TTL)
        env_prefix: Override env var prefix (default: uppercase name)

    Returns:
        TTLCache instance (shared singleton for this name)

    Example:
        cache = get_ttl_cache("scrape")  # Uses SCRAPE_CACHE_SIZE, SCRAPE_CACHE_TTL
        cache = get_ttl_cache("marker", maxsize=50, ttl=86400)  # Explicit config
    """
    if name in _caches:
        return _caches[name]

    # Determine env var prefix
    prefix = (env_prefix or name).upper()

    # Get config from env or defaults
    actual_maxsize = maxsize or int(os.getenv(f"{prefix}_CACHE_SIZE", "200"))
    actual_ttl = ttl or int(os.getenv(f"{prefix}_CACHE_TTL", "3600"))

    cache = TTLCache(maxsize=actual_maxsize, ttl=actual_ttl)
    _caches[name] = cache

    return cache


def clear_ttl_cache(name: str) -> bool:
    """Clear a named cache.

    Returns:
        True if cache existed and was cleared, False if not found.
    """
    if name in _caches:
        _caches[name].clear()
        return True
    return False


def get_ttl_cache_stats(name: str) -> dict[str, Any] | None:
    """Get statistics for a named cache.

    Returns:
        Dict with size, maxsize, ttl, or None if cache doesn't exist.
    """
    if name not in _caches:
        return None

    cache = _caches[name]
    return {
        "name": name,
        "size": len(cache),
        "maxsize": cache.maxsize,
        "ttl": cache.ttl,
    }
