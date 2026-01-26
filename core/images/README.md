# Images

Public domain image aggregator with LLM-powered selection across Pexels and Unsplash.

## Usage

### Basic Search

```python
from core.images import get_image

# Simple search returns first available result
result = await get_image("mountain landscape")
print(f"URL: {result.url}")
print(f"Photographer: {result.attribution.photographer}")
```

### LLM-Powered Selection

```python
# Use LLM to select best-fit image from multiple candidates
result = await get_image(
    "neural network",
    use_llm_selection=True,
    context="Article about deep learning architectures and gradient descent...",
    orientation="landscape"
)
```

### Custom Selection Criteria

```python
# Provide detailed criteria for more specific matching
result = await get_image(
    "team collaboration",
    use_llm_selection=True,
    context="Blog post about remote work best practices",
    custom_selection_criteria=(
        "Must show collaborative work in modern office setting, "
        "warm lighting, professional aesthetic, diverse team members"
    )
)
```

### Advanced Usage

```python
from core.images import ImageService, ImageSource, get_image_config

# Direct service access with custom configuration
service = ImageService(config=get_image_config())

# Search specific provider
result = await service.search(
    "sunset beach",
    preferred_provider=ImageSource.PEXELS,
    orientation="landscape"
)

# Clean up connections
await service.close()
```

## Input/Output

| Input | Type | Description |
|-------|------|-------------|
| `query` | `str` | Search term or phrase |
| `use_llm_selection` | `bool` | Enable LLM candidate selection (default: False) |
| `context` | `str \| None` | Document context for LLM selection |
| `orientation` | `str \| None` | "landscape", "portrait", or "square" |
| `custom_selection_criteria` | `str \| None` | Detailed selection requirements |

| Output | Type | Description |
|--------|------|-------------|
| `url` | `str` | Full-size image URL |
| `thumbnail_url` | `str \| None` | Thumbnail URL if available |
| `attribution` | `Attribution` | Photographer, license, source details |
| `metadata` | `ImageMetadata` | Dimensions, alt text, description |
| `provider` | `ImageSource` | Source provider (Pexels/Unsplash) |
| `query` | `str` | Original search term |

## Architecture

### Provider Fallback Chain

1. **Pexels** (primary) - Best free tier, no attribution required
2. **Unsplash** (fallback) - High quality, attribution required

The service automatically falls back to the next provider if:
- Provider not configured (missing API key)
- Rate limit exceeded
- No results found
- Provider error

### LLM Selection Mode

When `use_llm_selection=True`:

1. Searches all configured providers in parallel
2. Retrieves 3 candidates from each (6 total max)
3. LLM evaluates candidates based on context/criteria
4. Returns best-matching image with reasoning

Uses Haiku tier for cost-effective selection. Falls back to first result on LLM failure.

### Caching

All search results are cached for 30 days (configurable) using persistent cache. Cache keys include:
- Search query
- Orientation
- Provider preference
- LLM selection flag

### Error Handling

```python
from core.images import get_image, NoResultsError, ImageError

try:
    result = await get_image("obscure query term")
except NoResultsError as e:
    print(f"No images found: {e.message}")
    print(f"Provider: {e.provider}")
except ImageError as e:
    print(f"Image service error: {e.message}")
```

## Configuration

### Environment Variables

```bash
# Required (at least one)
PEXELS_API_KEY=your_pexels_key
UNSPLASH_API_KEY=your_unsplash_key

# Optional
IMAGE_CACHE_TTL_DAYS=30  # Cache duration (default: 30)
IMAGE_TIMEOUT=15         # HTTP timeout in seconds (default: 15)
```

### Custom Configuration

```python
from core.images import ImageConfig, ImageService

config = ImageConfig(
    pexels_api_key="your_key",
    unsplash_api_key="your_key",
    cache_ttl_days=7,
    timeout=10.0
)

service = ImageService(config=config)
```

## Related Modules

- `core.utils.async_http_client` - HTTP client with connection pooling
- `core.utils.caching` - Cache key generation utilities
- `workflows.shared.persistent_cache` - Persistent caching layer
- `workflows.shared.llm_utils` - LLM structured output utilities

## Extending Providers

Add new providers by implementing `BaseImageProvider`:

```python
from core.images.providers.base import BaseImageProvider
from core.images.types import ImageSource, ImageResult

class CustomProvider(BaseImageProvider):
    @property
    def source(self) -> ImageSource:
        return ImageSource.CUSTOM

    @property
    def is_available(self) -> bool:
        return bool(self._config.custom_api_key)

    async def search(
        self,
        query: str,
        limit: int = 5,
        orientation: str | None = None,
    ) -> list[ImageResult]:
        # Implementation
        pass
```

Register in `providers/__init__.py`:

```python
PROVIDER_REGISTRY[ImageSource.CUSTOM] = CustomProvider
```
