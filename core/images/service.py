"""Image search service with provider fallback."""

import asyncio
import logging

from core.utils.async_http_client import register_cleanup
from core.utils.caching import generate_cache_key
from workflows.shared.persistent_cache import get_cached, set_cached

from .config import ImageConfig, get_image_config
from .errors import NoResultsError, ProviderError, RateLimitError
from .providers import PROVIDER_REGISTRY, BaseImageProvider
from .selection import select_best_image
from .types import ImageResult, ImageSource

logger = logging.getLogger(__name__)


class ImageService:
    """Unified image search service with provider fallback.

    Fallback chain:
    1. Pexels (if configured) - best free tier, no attribution
    2. Unsplash (if configured) - high quality, requires attribution

    Usage:
        service = get_image_service()
        result = await service.search("sunset beach")

        # With LLM selection
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
        custom_selection_criteria: str | None = None,
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
            custom_selection_criteria: If provided, use these detailed criteria
                for LLM selection instead of the generic selection prompt.
                Enables more specific matching (e.g., "must show collaborative
                work in modern office, warm lighting, professional aesthetic").

        Returns:
            Best matching ImageResult

        Raises:
            NoResultsError: No images found across all providers
        """
        # Check cache first (include use_llm_selection in key since results differ)
        cache_key = generate_cache_key(
            "image_search",
            query,
            str(orientation or ""),
            str(preferred_provider or ""),
            str(use_llm_selection),
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

        # LLM selection: search all providers in parallel, combine results
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
            result = await select_best_image(
                all_results, query, context, custom_selection_criteria
            )
            set_cached("images", cache_key, result.model_dump())
            return result

        # Simple mode: sequential fallback, return first result
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


# Module singleton
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
