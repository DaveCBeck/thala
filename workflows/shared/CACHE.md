# Persistent Caching System

Hash-based persistent file caching for expensive operations to avoid redundant processing.

## Overview

This caching system provides:
- **Content-addressed storage** using SHA256 hashing
- **TTL-based expiration** (configurable per cache type)
- **Multiple serialization formats** (pickle, JSON)
- **Automatic directory management**
- **Cache statistics and cleanup utilities**

## Cache Types

| Type | Location | TTL | Purpose |
|------|----------|-----|---------|
| `openalex` | `.cache/openalex/` | 30 days | OpenAlex API metadata lookups |
| `marker` | `.cache/marker/` | 30 days | PDF to Markdown conversion results |
| `llm_results` | `.cache/llm_results/` | 7 days | Complete LLM response caching |

### OpenAlex API Cache

Caches:
- Work metadata lookups by DOI
- Forward/backward citation queries
- Author works queries
- DOI to OpenAlex ID resolution

### Marker PDF Processing Cache

Caches PDF conversion results using content hash:
- **Cache key**: `SHA256(file_content):quality:langs`
- **Benefit**: Skip GPU processing for already-processed files
- **Use case**: Re-running workflows, testing with same documents

### LLM Results Cache

Caches complete LLM responses for identical prompts:
- **Cache key**: `SHA256(model||system_prompt||user_prompt)`
- **Benefit**: 100% cost savings and instant results
- **Use case**: Development, testing, re-running workflows with same inputs

## Usage

### Automatic Caching (OpenAlex)

```python
from langchain_tools.openalex import get_work_by_doi

# First call: API fetch + cache
work = await get_work_by_doi("10.1234/example")

# Second call: cache hit
work = await get_work_by_doi("10.1234/example")  # <1ms
```

### Automatic Caching (Marker)

```python
from workflows.shared.marker_client import MarkerClient

async with MarkerClient() as client:
    # First call: GPU processing + cache
    result = await client.convert(
        file_path="paper.pdf",
        absolute_path="/full/path/to/paper.pdf"
    )

    # Second call: cache hit (even after service restart)
    result = await client.convert(
        file_path="paper.pdf",
        absolute_path="/full/path/to/paper.pdf"
    )
```

### Optional LLM Result Caching

```python
from workflows.shared.llm_utils import invoke_with_result_cache, get_llm, create_cached_messages

async def invoke_llm():
    llm = get_llm(ModelTier.SONNET)
    messages = create_cached_messages(system_prompt, user_prompt)
    return await llm.ainvoke(messages)

# Wraps existing LLM call with result caching
response = await invoke_with_result_cache(
    invoke_llm,
    system_prompt=SYSTEM_INSTRUCTIONS,
    user_prompt=document_text,
    model_name="sonnet",
    ttl_days=7,
)
```

### Manual Cache Operations

```python
from workflows.shared.persistent_cache import get_cached, set_cached

# Check cache
result = get_cached("my_type", "my_key", ttl_days=7)
if result is None:
    result = expensive_operation()
    set_cached("my_type", "my_key", result)
```

### Cache Management

```python
from workflows.shared.persistent_cache import clear_cache, get_cache_stats

# View statistics
stats = get_cache_stats()
# {'openalex': {'files': 150, 'size_mb': 5.2},
#  'marker': {'files': 20, 'size_mb': 45.3}}

# Clear specific cache
clear_cache("openalex")

# Clear all caches
clear_cache()
```

## Configuration

Set custom cache directory:

```bash
export THALA_CACHE_DIR=/custom/cache/path
```

Default: `/home/dave/thala/.cache`

## Performance Impact

### Before Caching
```
Paper pipeline: 14 acquired (0 from cache)
- OpenAlex lookups: 14 × 300ms = 4.2s
- PDF processing: 14 × 15s = 210s
Total: ~214s
```

### After Caching (Subsequent Runs)
```
Paper pipeline: 14 acquired (14 from cache)
- OpenAlex lookups: 14 × <1ms = ~14ms
- PDF processing: 14 × <100ms = ~1.4s
Total: ~1.5s
```

**Speedup**: ~140x faster on re-runs

### Expected Metrics

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| OpenAlex DOI lookup | 200-500ms | <1ms | 200-500x |
| Marker PDF processing | 5-30s | <100ms | 50-300x |
| LLM identical query | $0.XX + latency | 0ms + $0 | Instant |

## Cache Invalidation

Automatic invalidation when:
1. **TTL expires** (configurable per type)
2. **Manual clear** via `clear_cache()`
3. **Content changes** (for content-hashed caches)

## File Structure

```
.cache/
├── openalex/
│   ├── a1b2c3d4e5f6.pkl  # work:10.1234/example
│   ├── f7e8d9c0b1a2.pkl  # forward:10.1234/example:50:10:None
│   └── ...
├── marker/
│   ├── 1a2b3c4d5e6f.pkl  # SHA256(pdf_content):fast:English
│   └── ...
└── llm_results/
    ├── 9z8y7x6w5v4u.pkl  # SHA256(sonnet||system||user)
    └── ...
```

Filename format: `{SHA256(key)[:16]}.{pkl|json}`

## Testing

```bash
python3 testing/test_cache.py
```

## Notes

- Cache files use pickle by default for efficiency
- JSON format available via `format="json"` parameter
- Cache directory created automatically on first use
- Safe for concurrent access (atomic file operations)
- No cleanup required (TTL-based expiration)
