"""Persistent file-based caching for expensive operations."""

import hashlib
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("THALA_CACHE_DIR", "/home/dave/thala/.cache"))

# Global cache disable flag - set THALA_CACHE_DISABLED=1 to disable all caching
CACHE_DISABLED = os.getenv("THALA_CACHE_DISABLED", "").lower() in ("1", "true", "yes")


def _get_cache_path(cache_type: str, key: str, format: str = "pickle") -> Path:
    """Get cache file path for a given type and key."""
    cache_subdir = CACHE_DIR / cache_type
    cache_subdir.mkdir(parents=True, exist_ok=True)

    key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    ext = "pkl" if format == "pickle" else "json"
    return cache_subdir / f"{key_hash}.{ext}"


def _is_cache_valid(cache_path: Path, ttl_days: int) -> bool:
    """Check if cache file exists and is within TTL."""
    if not cache_path.exists():
        return False

    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - mtime
    return age < timedelta(days=ttl_days)


def get_cached(
    cache_type: str,
    key: str,
    ttl_days: int = 7,
    format: str = "pickle",
) -> Optional[Any]:
    """Get cached value if valid.

    Args:
        cache_type: Cache category (e.g., 'openalex', 'marker')
        key: Cache key (will be hashed)
        ttl_days: Time-to-live in days (default: 7)
        format: 'pickle' or 'json' (default: pickle)

    Returns:
        Cached value or None if not found/expired
    """
    if CACHE_DISABLED:
        return None

    cache_path = _get_cache_path(cache_type, key, format)

    if not _is_cache_valid(cache_path, ttl_days):
        return None

    try:
        if format == "json":
            with open(cache_path, "r") as f:
                return json.load(f)
        else:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to read cache {cache_path}: {e}")
        return None


def set_cached(
    cache_type: str,
    key: str,
    value: Any,
    format: str = "pickle",
) -> None:
    """Save value to cache.

    Args:
        cache_type: Cache category (e.g., 'openalex', 'marker')
        key: Cache key (will be hashed)
        value: Value to cache
        format: 'pickle' or 'json' (default: pickle)
    """
    if CACHE_DISABLED:
        return

    cache_path = _get_cache_path(cache_type, key, format)

    try:
        if format == "json":
            with open(cache_path, "w") as f:
                json.dump(value, f)
        else:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
        logger.debug(f"Cached to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to write cache {cache_path}: {e}")


def cached(
    cache_type: str,
    ttl_days: int = 7,
    format: str = "pickle",
    key_fn: Optional[Callable] = None,
):
    """Decorator to cache function results persistently.

    Args:
        cache_type: Cache category (e.g., 'openalex', 'marker')
        ttl_days: Time-to-live in days (default: 7)
        format: 'pickle' or 'json' (default: pickle)
        key_fn: Optional function to generate cache key from args

    Example:
        @cached(cache_type='openalex', ttl_days=30)
        async def get_work_by_doi(doi: str):
            ...
    """
    def decorator(fn):
        async def wrapper(*args, **kwargs):
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                cache_key = f"{fn.__name__}:{args}:{kwargs}"

            cached_value = get_cached(cache_type, cache_key, ttl_days, format)
            if cached_value is not None:
                logger.debug(f"Cache hit for {fn.__name__}")
                return cached_value

            result = await fn(*args, **kwargs)

            if result is not None:
                set_cached(cache_type, cache_key, result, format)

            return result
        return wrapper
    return decorator


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file contents.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of file hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def clear_cache(cache_type: Optional[str] = None) -> int:
    """Clear cached files.

    Args:
        cache_type: Specific cache type to clear, or None for all

    Returns:
        Number of files deleted
    """
    if cache_type:
        target_dir = CACHE_DIR / cache_type
        if not target_dir.exists():
            return 0
        dirs_to_clear = [target_dir]
    else:
        dirs_to_clear = [d for d in CACHE_DIR.iterdir() if d.is_dir()]

    count = 0
    for cache_dir in dirs_to_clear:
        for cache_file in cache_dir.glob("*"):
            if cache_file.is_file():
                cache_file.unlink()
                count += 1

    logger.info(f"Cleared {count} cached files")
    return count


def get_cache_stats(cache_type: Optional[str] = None) -> dict[str, Any]:
    """Get cache statistics.

    Args:
        cache_type: Specific cache type, or None for all

    Returns:
        Dict with cache stats
    """
    if cache_type:
        target_dirs = [CACHE_DIR / cache_type] if (CACHE_DIR / cache_type).exists() else []
    else:
        target_dirs = [d for d in CACHE_DIR.iterdir() if d.is_dir()]

    stats = {}
    for cache_dir in target_dirs:
        files = list(cache_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        stats[cache_dir.name] = {
            "files": len(files),
            "size_mb": total_size / (1024 * 1024),
        }

    return stats
