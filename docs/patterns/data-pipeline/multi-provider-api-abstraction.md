---
name: multi-provider-api-abstraction
title: "Multi-Provider API Abstraction with Fallback and Aggregation"
date: 2026-01-29
category: data-pipeline
applicability:
  - "Integrating multiple external APIs that provide similar functionality"
  - "When provider availability varies and graceful fallback is required"
  - "When combining results from multiple providers yields better outcomes"
components: [api_endpoint, async_task, configuration, llm_call]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [api-abstraction, provider-pattern, fallback-chain, parallel-aggregation, registry-pattern, llm-selection, lazy-initialization, caching]
---

# Multi-Provider API Abstraction with Fallback and Aggregation

## Intent

Provide a unified interface for searching across multiple external API providers with automatic fallback, optional parallel aggregation, and intelligent result selection, while isolating provider-specific implementation details behind a common abstraction.

## Motivation

When integrating multiple APIs that provide similar functionality (image search, payment processing, translation, etc.), several challenges arise:

1. **Provider diversity**: Each API has different authentication, request formats, and response structures
2. **Availability variance**: Some providers may be unavailable due to missing API keys or rate limits
3. **Quality variance**: Different providers excel at different queries
4. **Fallback needs**: When one provider fails, others should be tried automatically
5. **Result optimization**: Sometimes combining results from all providers yields better outcomes

This pattern addresses these challenges through:
- Abstract base class defining the provider contract
- Provider registry for dynamic provider management
- Service layer with two execution modes (fallback vs. aggregation)
- Optional LLM-based intelligent selection from aggregated results
- Result caching to reduce API costs

## Applicability

Use this pattern when:
- Integrating 2+ external APIs with similar functionality
- Provider availability may vary at runtime
- Graceful degradation is required when providers fail
- Combining results from multiple sources may yield better outcomes
- Provider-specific logic should be hidden from consumers

Do NOT use this pattern when:
- Only one provider will ever be used
- Providers have fundamentally different capabilities (not substitutable)
- Real-time requirements prevent fallback chains
- Result format differs significantly between providers (no common abstraction possible)

## Structure

```
                        +------------------+
                        |   Consumer Code  |
                        +--------+---------+
                                 |
                        +--------v---------+
                        |  ImageService    |
                        |  (unified API)   |
                        +--------+---------+
                                 |
           +---------------------+---------------------+
           |                                           |
  +--------v---------+                        +--------v---------+
  | Simple Mode      |                        | Aggregation Mode |
  | (fallback chain) |                        | (parallel search)|
  +--------+---------+                        +--------+---------+
           |                                           |
           |    Priority Order                         |    asyncio.gather
           |    Pexels -> Unsplash                     |    All Providers
           |                                           |
           v                                  +--------v---------+
  First successful result                     | select_best_image|
                                              | (LLM selection)  |
                                              +------------------+

Provider Layer:
+------------------+     +------------------+     +------------------+
| BaseImageProvider|<----| PexelsProvider   |     | UnsplashProvider |
| (ABC)            |     +------------------+     +------------------+
+------------------+             ^                        ^
| + search()       |             |                        |
| + source         |     +-------+------------------------+
| + is_available   |     |
| + close()        |     |  PROVIDER_REGISTRY
+------------------+     |  {PEXELS: PexelsProvider, UNSPLASH: UnsplashProvider}
```

**Mode Comparison:**

| Aspect | Simple (Fallback) | Aggregation (LLM Selection) |
|--------|-------------------|----------------------------|
| Speed | Fast (stops at first success) | Slower (queries all providers) |
| Cost | Lower (fewer API calls) | Higher (all providers + LLM) |
| Quality | First available result | Best match from all candidates |
| Use case | General search | Context-specific selection |

## Implementation

### Step 1: Define Unified Types

Create common types that abstract provider-specific differences:

```python
# core/images/types.py
from enum import Enum
from pydantic import BaseModel, Field


class ImageSource(str, Enum):
    """Supported image providers."""
    PEXELS = "pexels"
    UNSPLASH = "unsplash"


class Attribution(BaseModel):
    """Attribution information for image usage."""
    required: bool = False
    photographer: str | None = None
    photographer_url: str | None = None
    source: ImageSource
    source_url: str = Field(description="Link to original on source site")
    license: str = "Various"


class ImageMetadata(BaseModel):
    """Image metadata."""
    width: int
    height: int
    alt_text: str | None = None
    description: str | None = None


class ImageResult(BaseModel):
    """Unified result from any image provider."""
    url: str = Field(description="Full-size image URL")
    thumbnail_url: str | None = None
    attribution: Attribution
    metadata: ImageMetadata
    provider: ImageSource
    query: str = Field(description="Original search term")
```

### Step 2: Define the Provider Contract

Abstract base class establishes the interface all providers must implement:

```python
# core/images/providers/base.py
from abc import ABC, abstractmethod
import httpx

from ..config import ImageConfig
from ..types import ImageResult, ImageSource


class BaseImageProvider(ABC):
    """Abstract base for image providers."""

    def __init__(self, config: ImageConfig):
        self._config = config
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client (lazy initialization)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._config.timeout)
        return self._client

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 5,
        orientation: str | None = None,
    ) -> list[ImageResult]:
        """Search for images.

        Args:
            query: Search term
            limit: Maximum results to return
            orientation: "landscape", "portrait", or "square"

        Returns:
            List of ImageResult objects
        """
        pass

    @property
    @abstractmethod
    def source(self) -> ImageSource:
        """Provider source identifier."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether provider is configured and available."""
        pass

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
```

Key design decisions:
- **Lazy HTTP client**: Created on first use, not at initialization
- **Shared cleanup**: Base class handles resource cleanup
- **`is_available` property**: Allows runtime availability checking
- **`source` property**: Enables result attribution

### Step 3: Implement Concrete Providers

Each provider implements the interface with API-specific logic:

```python
# core/images/providers/pexels.py
from ..errors import ProviderError, RateLimitError
from ..types import Attribution, ImageMetadata, ImageResult, ImageSource
from .base import BaseImageProvider

BASE_URL = "https://api.pexels.com/v1"


class PexelsProvider(BaseImageProvider):
    """Pexels image provider."""

    @property
    def source(self) -> ImageSource:
        return ImageSource.PEXELS

    @property
    def is_available(self) -> bool:
        return self._config.pexels_available

    async def search(
        self,
        query: str,
        limit: int = 5,
        orientation: str | None = None,
    ) -> list[ImageResult]:
        """Search Pexels for images."""
        if not self.is_available:
            raise ProviderError("Pexels API key not configured", provider="pexels")

        client = await self._get_client()

        params: dict = {"query": query, "per_page": limit}
        if orientation:
            params["orientation"] = orientation

        try:
            response = await client.get(
                f"{BASE_URL}/search",
                params=params,
                headers={"Authorization": self._config.pexels_api_key},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Pexels rate limit exceeded", provider="pexels")
            raise ProviderError(f"Pexels API error: {e}", provider="pexels")

        data = response.json()
        return [self._parse_photo(photo, query) for photo in data.get("photos", [])]

    def _parse_photo(self, photo: dict, query: str) -> ImageResult:
        """Parse Pexels API response into ImageResult."""
        return ImageResult(
            url=photo["src"]["large2x"],
            thumbnail_url=photo["src"]["medium"],
            attribution=Attribution(
                required=False,  # Pexels doesn't require attribution
                photographer=photo.get("photographer"),
                photographer_url=photo.get("photographer_url"),
                source=ImageSource.PEXELS,
                source_url=photo["url"],
                license="Pexels License",
            ),
            metadata=ImageMetadata(
                width=photo["width"],
                height=photo["height"],
                alt_text=photo.get("alt"),
            ),
            provider=ImageSource.PEXELS,
            query=query,
        )
```

```python
# core/images/providers/unsplash.py
class UnsplashProvider(BaseImageProvider):
    """Unsplash image provider."""

    @property
    def source(self) -> ImageSource:
        return ImageSource.UNSPLASH

    @property
    def is_available(self) -> bool:
        return self._config.unsplash_available

    async def search(
        self,
        query: str,
        limit: int = 5,
        orientation: str | None = None,
    ) -> list[ImageResult]:
        """Search Unsplash for images."""
        if not self.is_available:
            raise ProviderError("Unsplash API key not configured", provider="unsplash")

        client = await self._get_client()

        params: dict = {"query": query, "per_page": limit}
        if orientation:
            params["orientation"] = orientation

        try:
            response = await client.get(
                f"{BASE_URL}/search/photos",
                params=params,
                headers={"Authorization": f"Client-ID {self._config.unsplash_api_key}"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 403):
                raise RateLimitError("Unsplash rate limit exceeded", provider="unsplash")
            raise ProviderError(f"Unsplash API error: {e}", provider="unsplash")

        data = response.json()
        return [self._parse_photo(photo, query) for photo in data.get("results", [])]

    def _parse_photo(self, photo: dict, query: str) -> ImageResult:
        """Parse Unsplash API response into ImageResult."""
        user = photo.get("user", {})
        urls = photo.get("urls", {})

        return ImageResult(
            url=urls.get("regular", urls.get("full", "")),
            thumbnail_url=urls.get("small"),
            attribution=Attribution(
                required=True,  # Unsplash requires attribution
                photographer=user.get("name"),
                photographer_url=user.get("links", {}).get("html"),
                source=ImageSource.UNSPLASH,
                source_url=photo.get("links", {}).get("html", ""),
                license="Unsplash License",
            ),
            metadata=ImageMetadata(
                width=photo.get("width", 0),
                height=photo.get("height", 0),
                alt_text=photo.get("alt_description"),
                description=photo.get("description"),
            ),
            provider=ImageSource.UNSPLASH,
            query=query,
        )
```

### Step 4: Create the Provider Registry

Registry pattern enables dynamic provider lookup and easy extensibility:

```python
# core/images/providers/__init__.py
from ..types import ImageSource
from .base import BaseImageProvider
from .pexels import PexelsProvider
from .unsplash import UnsplashProvider

PROVIDER_REGISTRY: dict[ImageSource, type[BaseImageProvider]] = {
    ImageSource.PEXELS: PexelsProvider,
    ImageSource.UNSPLASH: UnsplashProvider,
}

__all__ = [
    "BaseImageProvider",
    "PexelsProvider",
    "UnsplashProvider",
    "PROVIDER_REGISTRY",
]
```

Adding a new provider requires:
1. Create new provider class implementing `BaseImageProvider`
2. Add enum value to `ImageSource`
3. Register in `PROVIDER_REGISTRY`
4. Add config for API key availability check

### Step 5: Implement LLM-Based Selection

For aggregation mode, use a lightweight LLM to pick the best result:

```python
# core/images/selection.py
from pydantic import BaseModel, Field

from workflows.shared.llm_utils import ModelTier, get_structured_output
from .types import ImageResult


class ImageSelection(BaseModel):
    """LLM output for image selection."""
    selected_index: int = Field(description="0-based index of best image")
    reasoning: str = Field(description="Brief explanation of selection")


SELECTION_SYSTEM = """You are an image selector for article headers.
Select the image that best complements the article context without being too literal.
Consider composition, mood, and thematic relevance."""

SELECTION_USER = """Select the best image for the given context.

Context: {context}

Search query: {query}

Image candidates:
{candidates}

Choose the image index (0-based) that best matches the context."""


async def select_best_image(
    candidates: list[ImageResult],
    query: str,
    context: str,
) -> ImageResult:
    """Use LLM to select best image from candidates.

    Args:
        candidates: List of image results to choose from
        query: Original search query
        context: Article/document context

    Returns:
        Best matching ImageResult
    """
    if len(candidates) <= 1:
        return candidates[0]

    # Format candidates for LLM
    candidate_text = "\n".join(
        f"{i}. [{c.provider.value}] "
        f"{c.metadata.alt_text or c.metadata.description or 'No description'} "
        f"({c.metadata.width}x{c.metadata.height})"
        for i, c in enumerate(candidates)
    )

    try:
        result = await get_structured_output(
            output_schema=ImageSelection,
            user_prompt=SELECTION_USER.format(
                context=context[:2000],  # Truncate long context
                query=query,
                candidates=candidate_text,
            ),
            system_prompt=SELECTION_SYSTEM,
            tier=ModelTier.HAIKU,  # Cheapest tier for simple selection
        )

        index = max(0, min(result.selected_index, len(candidates) - 1))
        return candidates[index]

    except Exception as e:
        # Graceful fallback on LLM error
        logger.warning(f"LLM selection failed, using first result: {e}")
        return candidates[0]
```

Key decisions:
- **Cheapest model tier**: Simple selection task doesn't need powerful models
- **Graceful fallback**: LLM failure returns first result, not error
- **Context truncation**: Prevents token limit issues
- **Structured output**: Pydantic model ensures valid response format

### Step 6: Implement the Service Layer

The service provides two execution modes through a unified interface:

```python
# core/images/service.py
import asyncio
import logging

from core.utils.async_http_client import register_cleanup
from workflows.shared.persistent_cache import get_cached, set_cached

from .config import ImageConfig, get_image_config
from .errors import NoResultsError, ProviderError, RateLimitError
from .providers import PROVIDER_REGISTRY, BaseImageProvider
from .selection import select_best_image
from .types import ImageResult, ImageSource

logger = logging.getLogger(__name__)


class ImageService:
    """Unified image search service with provider fallback.

    Fallback chain priority:
    1. Pexels (if configured) - best free tier, no attribution required
    2. Unsplash (if configured) - high quality, requires attribution

    Usage:
        service = get_image_service()
        result = await service.search("sunset beach")

        # With LLM selection (aggregates from all providers)
        result = await service.search(
            "algorithm visualization",
            use_llm_selection=True,
            context="Article about machine learning..."
        )
    """

    def __init__(self, config: ImageConfig | None = None):
        self._config = config or get_image_config()
        self._providers: dict[ImageSource, BaseImageProvider] = {}

    def _get_provider(self, source: ImageSource) -> BaseImageProvider:
        """Get or create provider (lazy initialization)."""
        if source not in self._providers:
            provider_class = PROVIDER_REGISTRY[source]
            self._providers[source] = provider_class(self._config)
        return self._providers[source]

    def _get_priority_order(self) -> list[ImageSource]:
        """Get providers in priority order based on availability."""
        order = []
        if self._config.pexels_available:
            order.append(ImageSource.PEXELS)
        if self._config.unsplash_available:
            order.append(ImageSource.UNSPLASH)
        return order

    async def _search_provider(
        self,
        source: ImageSource,
        query: str,
        limit: int,
        orientation: str | None,
    ) -> list[ImageResult]:
        """Search a single provider, returning empty list on failure."""
        try:
            provider = self._get_provider(source)
            if not provider.is_available:
                return []
            results = await provider.search(query, limit=limit, orientation=orientation)
            logger.debug(f"Got {len(results)} results from {source.value}")
            return results
        except (RateLimitError, ProviderError) as e:
            logger.debug(f"Provider {source.value} failed: {e}")
            return []

    async def search(
        self,
        query: str,
        use_llm_selection: bool = False,
        context: str | None = None,
        orientation: str | None = None,
        preferred_provider: ImageSource | None = None,
    ) -> ImageResult:
        """Search for best-fit image across providers.

        Args:
            query: Search term
            use_llm_selection: Use LLM to select best image from candidates.
                When True, searches all available providers in parallel and
                uses LLM to pick the best match from combined results.
            context: Article/document context for LLM selection
            orientation: "landscape", "portrait", or "square"
            preferred_provider: Force specific provider (skip fallback)

        Returns:
            Best matching ImageResult

        Raises:
            NoResultsError: No images found across all providers
        """
        # Check cache first
        cache_key = generate_cache_key(
            "image_search", query, str(orientation or ""),
            str(preferred_provider or ""), str(use_llm_selection),
        )
        cached = get_cached("images", cache_key, ttl_days=self._config.cache_ttl_days)
        if cached:
            logger.debug(f"Cache hit for image search: {query}")
            return ImageResult.model_validate(cached)

        # Determine which providers to use
        providers = (
            [preferred_provider] if preferred_provider else self._get_priority_order()
        )

        if not providers:
            raise NoResultsError(
                "No image providers configured. Set PEXELS_API_KEY or UNSPLASH_API_KEY.",
                provider="none",
            )

        # === Aggregation Mode: Parallel search + LLM selection ===
        if use_llm_selection and context and len(providers) > 1:
            tasks = [
                self._search_provider(source, query, limit=3, orientation=orientation)
                for source in providers
            ]
            results_lists = await asyncio.gather(*tasks)
            all_results = [r for results in results_lists for r in results]

            if not all_results:
                raise NoResultsError(
                    f"No images found for '{query}' across all providers",
                    provider="all",
                )

            logger.info(
                f"LLM selecting from {len(all_results)} candidates across "
                f"{sum(1 for r in results_lists if r)} providers"
            )
            result = await select_best_image(all_results, query, context)
            set_cached("images", cache_key, result.model_dump())
            return result

        # === Simple Mode: Sequential fallback ===
        for source in providers:
            results = await self._search_provider(
                source, query, limit=1, orientation=orientation
            )
            if results:
                result = results[0]
                set_cached("images", cache_key, result.model_dump())
                logger.info(f"Found image for '{query}' from {source.value}")
                return result

        raise NoResultsError(
            f"No images found for '{query}' across all providers",
            provider="all",
        )

    async def close(self) -> None:
        """Close all provider connections."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()


# Module singleton with cleanup registration
_service: ImageService | None = None


def get_image_service() -> ImageService:
    """Get global ImageService instance."""
    global _service
    if _service is None:
        _service = ImageService()
        register_cleanup("ImageService", _close_image_service)
    return _service


async def _close_image_service() -> None:
    """Close the global ImageService."""
    global _service
    if _service:
        await _service.close()
        _service = None
```

### Step 7: Define Error Hierarchy

Custom exceptions carry provider context:

```python
# core/images/errors.py
class ImageError(Exception):
    """Base image service exception."""

    def __init__(self, message: str, provider: str | None = None):
        self.message = message
        self.provider = provider
        super().__init__(message)


class NoResultsError(ImageError):
    """No images found for query."""
    pass


class RateLimitError(ImageError):
    """API rate limit exceeded."""
    pass


class ProviderError(ImageError):
    """Provider-specific error."""
    pass
```

### Step 8: Configuration Management

Environment-based configuration with availability checks:

```python
# core/images/config.py
from dataclasses import dataclass, field
import os


@dataclass
class ImageConfig:
    """Configuration for image search service."""

    pexels_api_key: str | None = field(
        default_factory=lambda: os.environ.get("PEXELS_API_KEY")
    )
    unsplash_api_key: str | None = field(
        default_factory=lambda: os.environ.get("UNSPLASH_API_KEY")
    )
    cache_ttl_days: int = field(
        default_factory=lambda: int(os.environ.get("IMAGE_CACHE_TTL_DAYS", "30"))
    )
    timeout: float = field(
        default_factory=lambda: float(os.environ.get("IMAGE_TIMEOUT", "15"))
    )

    @property
    def pexels_available(self) -> bool:
        """Check if Pexels provider is configured."""
        return bool(self.pexels_api_key)

    @property
    def unsplash_available(self) -> bool:
        """Check if Unsplash provider is configured."""
        return bool(self.unsplash_api_key)
```

### Step 9: Public Module Interface

Clean exports for consumers:

```python
# core/images/__init__.py
"""Public domain image aggregator.

Example:
    from core.images import get_image

    # Simple search (fallback mode)
    result = await get_image("mountain landscape")

    # With LLM selection (aggregation mode)
    result = await get_image(
        "neural network",
        use_llm_selection=True,
        context="Article about deep learning..."
    )
"""

from .service import ImageService, get_image_service
from .types import Attribution, ImageMetadata, ImageResult, ImageSource


async def get_image(
    query: str,
    use_llm_selection: bool = False,
    context: str | None = None,
    orientation: str | None = None,
) -> ImageResult:
    """Search for a public domain image."""
    service = get_image_service()
    return await service.search(
        query,
        use_llm_selection=use_llm_selection,
        context=context,
        orientation=orientation,
    )
```

## When to Use Each Mode

### Simple Mode (Sequential Fallback)

Best for:
- General-purpose image search
- When any relevant image is acceptable
- When minimizing API calls matters
- When LLM cost is a concern

```python
# Returns first successful result from priority-ordered providers
result = await get_image("sunset beach")
```

### Aggregation Mode (Parallel + LLM Selection)

Best for:
- Content creation with specific requirements
- When image-context fit matters
- When multiple providers offer meaningfully different results
- When quality trumps cost

```python
# Searches all providers, uses LLM to pick best match for context
result = await get_image(
    "abstract technology",
    use_llm_selection=True,
    context="Article about quantum computing breakthroughs..."
)
```

## Adding a New Provider

1. **Add enum value**:
```python
class ImageSource(str, Enum):
    PEXELS = "pexels"
    UNSPLASH = "unsplash"
    PIXABAY = "pixabay"  # New
```

2. **Create provider class**:
```python
class PixabayProvider(BaseImageProvider):
    @property
    def source(self) -> ImageSource:
        return ImageSource.PIXABAY

    @property
    def is_available(self) -> bool:
        return self._config.pixabay_available

    async def search(self, query: str, limit: int = 5, ...) -> list[ImageResult]:
        # Implement Pixabay-specific API call
        ...
```

3. **Register in registry**:
```python
PROVIDER_REGISTRY: dict[ImageSource, type[BaseImageProvider]] = {
    ImageSource.PEXELS: PexelsProvider,
    ImageSource.UNSPLASH: UnsplashProvider,
    ImageSource.PIXABAY: PixabayProvider,  # New
}
```

4. **Add config property**:
```python
@property
def pixabay_available(self) -> bool:
    return bool(os.environ.get("PIXABAY_API_KEY"))
```

## Consequences

### Benefits

- **Provider independence**: Business logic doesn't depend on specific APIs
- **Easy extensibility**: Adding providers is mechanical, not architectural
- **Graceful degradation**: Service works with any available subset of providers
- **Flexible execution**: Choose between speed (fallback) and quality (aggregation)
- **Cost optimization**: Caching reduces API calls; cheapest LLM tier for selection
- **Resource efficiency**: Lazy initialization avoids unused provider overhead

### Trade-offs

- **Abstraction overhead**: Common interface may not expose provider-unique features
- **LLM cost for aggregation**: Each aggregation mode search incurs LLM API cost
- **Cache key complexity**: Must include all parameters affecting result
- **Singleton state**: Shared service aids caching but complicates testing

### Async Considerations

- **`asyncio.gather` for parallel**: Standard pattern for concurrent provider queries
- **Lazy HTTP clients**: Created per-provider on first use
- **Cleanup registration**: Ensures proper resource cleanup on shutdown

## Related Patterns

- [Unified Scraping Service with Fallback Chain](./unified-scraping-service-fallback-chain.md) - Similar fallback pattern for scraping
- [Parallel AI Search Integration](./parallel-ai-search-integration.md) - Parallel search without abstraction layer
- [Hash-Based Persistent Caching](./hash-based-persistent-caching.md) - Caching strategy used here

## Known Uses in Thala

- `core/images/__init__.py`: Public API and `get_image()` function
- `core/images/service.py`: ImageService with fallback/aggregation modes
- `core/images/providers/base.py`: BaseImageProvider abstract base class
- `core/images/providers/pexels.py`: Pexels provider implementation
- `core/images/providers/unsplash.py`: Unsplash provider implementation
- `core/images/providers/__init__.py`: Provider registry
- `core/images/selection.py`: LLM-based image selection
- `core/images/types.py`: Unified type definitions
- `core/images/config.py`: Configuration management
- `core/images/errors.py`: Error hierarchy

## References

- [Pexels API Documentation](https://www.pexels.com/api/documentation/)
- [Unsplash API Documentation](https://unsplash.com/documentation)
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy) - Related pattern
- [Registry Pattern](https://martinfowler.com/eaaCatalog/registry.html) - Pattern used for provider lookup
