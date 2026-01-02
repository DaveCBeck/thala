"""Caching utilities for async functions."""

import hashlib
import logging
from functools import wraps
from typing import Callable, Optional, Any

from workflows.shared.persistent_cache import get_cached, set_cached

logger = logging.getLogger(__name__)


def generate_cache_key(*parts: str) -> str:
    """Generate a SHA256 cache key from parts."""
    combined = ":".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def async_cached(namespace: str, ttl_days: int = 7, key_fn: Optional[Callable] = None):
    """
    Decorator for caching async function results.

    Args:
        namespace: Cache namespace (e.g., 'embeddings', 'translation_server')
        ttl_days: Time-to-live in days
        key_fn: Optional function to generate cache key from args/kwargs
    """
    def decorator(fn: Callable):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                cache_key = generate_cache_key(fn.__name__, str(args), str(kwargs))

            cached_value = get_cached(namespace, cache_key, ttl_days=ttl_days)
            if cached_value is not None:
                logger.debug(f"Cache hit for {fn.__name__}")
                return cached_value

            result = await fn(*args, **kwargs)

            if result is not None:
                set_cached(namespace, cache_key, result)

            return result
        return wrapper
    return decorator
