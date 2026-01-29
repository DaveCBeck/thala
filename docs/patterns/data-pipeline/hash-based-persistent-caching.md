---
name: hash-based-persistent-caching
title: Hash-Based Persistent Caching Architecture
date: 2026-01-02
category: data-pipeline
applicability:
  - "Expensive API calls that return stable data"
  - "Document processing with deterministic outputs"
  - "Embedding generation from text content"
  - "Search results that can be reused within TTL windows"
components: [persistent_cache, openalex, marker, embeddings]
complexity: medium
verified_in_production: true
tags: [caching, persistence, ttl, api-optimization, cost-reduction, decorator]
---

# Hash-Based Persistent Caching Architecture

## Intent

Reduce API costs and processing time by caching expensive operations (API calls, document processing, embeddings) to disk with content-based keys and category-specific TTLs.

## Problem

Research workflows make many expensive external calls:
- **OpenAlex API**: DOI lookups, citation fetching, search queries
- **Marker service**: PDF to markdown conversion (GPU-intensive)
- **Embedding APIs**: Text vectorization (token costs)
- **Search APIs**: Perplexity, web search (per-query costs)

Without caching:
- Same DOI looked up multiple times across workflow runs
- Re-processing identical PDFs wastes GPU time
- Embedding identical text chunks repeatedly
- Search queries repeated unnecessarily

## Solution

Implement a file-based persistent cache with:
1. **Category-specific TTLs**: Different data types have different staleness tolerances
2. **Content-based hashing**: Cache keys derived from input content, not arbitrary IDs
3. **Decorator pattern**: Easy integration with existing async functions
4. **Format flexibility**: Pickle for complex objects, JSON for inspectable data

## Structure

```
workflows/shared/
├── persistent_cache.py    # Core caching implementation
│   ├── get_cached()       # Retrieve from cache with TTL check
│   ├── set_cached()       # Store to cache
│   ├── @cached            # Decorator for async functions
│   ├── compute_file_hash()# Content-based key generation
│   └── get_cache_stats()  # Monitoring utilities

.thala/cache/              # Cache storage directory
├── openalex/              # DOI lookups, citations (30-day TTL)
├── marker/                # PDF processing results (30-day TTL)
├── embeddings/            # Text embeddings (90-day TTL)
├── perplexity/            # Search results (7-day TTL)
└── translation_server/    # Translation results (30-day TTL)
```

## Implementation

### Core Cache Functions

```python
# workflows/shared/persistent_cache.py

import hashlib
import pickle
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".thala/cache")

# Category-specific TTLs (days)
DEFAULT_TTLS = {
    "openalex": 30,       # API data stable for a month
    "marker": 30,         # PDF processing deterministic
    "embeddings": 90,     # Embeddings very stable
    "perplexity": 7,      # Search results fresher
    "translation_server": 30,
}


def _get_cache_path(cache_type: str, key: str, format: str = "pickle") -> Path:
    """Generate cache file path from type and key."""
    # Hash the key to create safe filename
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    ext = "pkl" if format == "pickle" else "json"
    return CACHE_DIR / cache_type / f"{key_hash}.{ext}"


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
    ttl_days: int | None = None,
    format: str = "pickle",
) -> Optional[Any]:
    """Retrieve value from cache if valid.

    Args:
        cache_type: Category (openalex, marker, embeddings, etc.)
        key: Cache key (will be hashed)
        ttl_days: Override default TTL for this category
        format: "pickle" or "json"

    Returns:
        Cached value or None if not found/expired
    """
    ttl = ttl_days or DEFAULT_TTLS.get(cache_type, 7)
    cache_path = _get_cache_path(cache_type, key, format)

    if not _is_cache_valid(cache_path, ttl):
        return None

    try:
        if format == "pickle":
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        else:
            with open(cache_path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Cache read failed for {cache_type}/{key}: {e}")
        return None


def set_cached(
    cache_type: str,
    key: str,
    value: Any,
    format: str = "pickle",
) -> None:
    """Store value in cache.

    Args:
        cache_type: Category (openalex, marker, embeddings, etc.)
        key: Cache key (will be hashed)
        value: Value to cache
        format: "pickle" or "json"
    """
    cache_path = _get_cache_path(cache_type, key, format)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "pickle":
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
        else:
            with open(cache_path, "w") as f:
                json.dump(value, f)
        logger.debug(f"Cached {cache_type}/{key[:32]}...")
    except Exception as e:
        logger.warning(f"Cache write failed for {cache_type}/{key}: {e}")
```

### Decorator Pattern

```python
def cached(
    cache_type: str,
    ttl_days: int | None = None,
    key_func: Callable[..., str] | None = None,
    format: str = "pickle",
):
    """Decorator for caching async function results.

    Args:
        cache_type: Category for this function's cache
        ttl_days: Override default TTL
        key_func: Custom function to generate cache key from args
        format: "pickle" or "json"

    Usage:
        @cached(cache_type='openalex', ttl_days=30)
        async def get_work_by_doi(doi: str) -> dict:
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default: hash all arguments
                key = f"{func.__name__}:{args}:{kwargs}"

            # Check cache
            cached_value = get_cached(cache_type, key, ttl_days, format)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_type}/{key[:32]}...")
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                set_cached(cache_type, key, result, format)

            return result
        return wrapper
    return decorator
```

### Content-Based Hashing

```python
def compute_file_hash(file_path: str | Path) -> str:
    """Compute SHA256 hash of file content.

    Used for caching Marker results by PDF content, not filename.
    This ensures identical PDFs share cache entries even with
    different filenames.
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def compute_text_hash(text: str) -> str:
    """Compute hash of text content for embedding cache."""
    return hashlib.sha256(text.encode()).hexdigest()
```

### Cache Management Utilities

```python
def get_cache_stats() -> dict[str, dict]:
    """Get statistics about cache usage by category."""
    stats = {}

    if not CACHE_DIR.exists():
        return stats

    for category_dir in CACHE_DIR.iterdir():
        if not category_dir.is_dir():
            continue

        files = list(category_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        stats[category_dir.name] = {
            "files": len(files),
            "size_mb": total_size / (1024 * 1024),
            "oldest": min((f.stat().st_mtime for f in files), default=None),
            "newest": max((f.stat().st_mtime for f in files), default=None),
        }

    return stats


def clear_cache(cache_type: str | None = None, older_than_days: int | None = None) -> int:
    """Clear cache files.

    Args:
        cache_type: Specific category to clear (None = all)
        older_than_days: Only clear files older than this

    Returns:
        Number of files deleted
    """
    deleted = 0

    if cache_type:
        dirs = [CACHE_DIR / cache_type]
    else:
        dirs = [d for d in CACHE_DIR.iterdir() if d.is_dir()]

    cutoff = datetime.now() - timedelta(days=older_than_days) if older_than_days else None

    for cache_dir in dirs:
        for cache_file in cache_dir.glob("*"):
            if cutoff:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if mtime > cutoff:
                    continue
            cache_file.unlink()
            deleted += 1

    return deleted
```

## Usage

### Basic Caching with Decorator

```python
from workflows.shared.persistent_cache import cached

@cached(cache_type='openalex', ttl_days=30)
async def get_work_by_doi(doi: str) -> dict | None:
    """Fetch work metadata from OpenAlex API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.openalex.org/works/doi:{doi}")
        if response.status_code == 200:
            return response.json()
        return None


# First call: hits API, caches result
work = await get_work_by_doi("10.1234/example")

# Second call within 30 days: returns cached result
work = await get_work_by_doi("10.1234/example")
```

### Content-Based Caching for Files

```python
from workflows.shared.persistent_cache import get_cached, set_cached, compute_file_hash

async def process_pdf_with_marker(pdf_path: str) -> dict:
    """Process PDF with content-based caching."""
    # Use file content hash as key (not filename)
    content_hash = compute_file_hash(pdf_path)

    # Check cache
    cached_result = get_cached("marker", content_hash)
    if cached_result:
        logger.info(f"Marker cache hit for {pdf_path}")
        return cached_result

    # Process and cache
    result = await marker_client.convert(pdf_path)
    set_cached("marker", content_hash, result)

    return result
```

### Custom Key Functions

```python
@cached(
    cache_type='embeddings',
    ttl_days=90,
    key_func=lambda text, model: f"{model}:{compute_text_hash(text)}",
)
async def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get embedding with model-aware caching."""
    return await embedding_client.embed(text, model=model)
```

### Cache Statistics

```python
from workflows.shared.persistent_cache import get_cache_stats, clear_cache

# Check cache usage
stats = get_cache_stats()
for category, info in stats.items():
    print(f"{category}: {info['files']} files, {info['size_mb']:.1f} MB")

# Clear old entries
deleted = clear_cache(older_than_days=60)
print(f"Cleared {deleted} old cache files")

# Clear specific category
deleted = clear_cache(cache_type='perplexity')
```

## Guidelines

### TTL Selection by Data Type

| Category | TTL | Rationale |
|----------|-----|-----------|
| `openalex` | 30 days | Academic metadata rarely changes |
| `marker` | 30 days | PDF conversion deterministic |
| `embeddings` | 90 days | Same text = same embedding |
| `perplexity` | 7 days | Search results need freshness |
| `translation_server` | 30 days | Translation stable |

### Cache Key Best Practices

1. **APIs with stable IDs**: Use the ID directly (DOI, ISBN)
2. **Content processing**: Use content hash, not filename
3. **Multi-param functions**: Include all params that affect output
4. **Model-dependent**: Include model name/version in key

### Format Selection

| Format | Use When |
|--------|----------|
| `pickle` | Complex Python objects, nested dicts, embeddings |
| `json` | Need human inspection, simple data, debugging |

### Memory Considerations

- Pickle files can be large for embeddings/documents
- Monitor `.thala/cache/` size periodically
- Use `clear_cache(older_than_days=N)` in maintenance scripts

## Known Uses

- `workflows/research/subgraphs/academic_lit_review/paper_processor/` - OpenAlex caching
- `workflows/shared/marker_client.py` - PDF processing cache
- `workflows/shared/embedding_utils.py` - Embedding cache
- `workflows/research/nodes/search_perplexity.py` - Search cache

## Consequences

### Benefits
- **Cost reduction**: Avoid duplicate API calls and processing
- **Speed improvement**: Cache hits return instantly
- **Deterministic**: Same input always returns same cached output
- **Persistence**: Cache survives process restarts

### Trade-offs
- **Disk space**: Cache files accumulate over time
- **Staleness risk**: Data may become outdated within TTL
- **Complexity**: Cache invalidation logic needed for updates

## Related Patterns

- [Phased Pipeline Architecture](./phased-pipeline-architecture-gpu-queue.md) - Uses ES cache check
- [Prompt Caching Patterns](../llm-interaction/prompt-caching-patterns.md) - In-memory caching for prompts

## References

- [Python functools.wraps](https://docs.python.org/3/library/functools.html#functools.wraps)
- [hashlib](https://docs.python.org/3/library/hashlib.html)
