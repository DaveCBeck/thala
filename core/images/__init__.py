"""Public domain image aggregator.

Search for public domain images across Pexels and Unsplash.

Example:
    from core.images import get_image

    # Simple search
    result = await get_image("mountain landscape")
    print(result.url)

    # With LLM selection for better context matching
    result = await get_image(
        "neural network",
        use_llm_selection=True,
        context="Article about deep learning architectures..."
    )

Environment Variables:
    PEXELS_API_KEY: API key for Pexels (primary provider)
    UNSPLASH_ACCESS_KEY: Access key for Unsplash (fallback provider)
"""

from .config import ImageConfig, get_image_config
from .errors import ImageError, NoResultsError, ProviderError, RateLimitError
from .service import ImageService, get_image_service
from .types import Attribution, ImageMetadata, ImageResult, ImageSource


async def get_image(
    query: str,
    use_llm_selection: bool = False,
    context: str | None = None,
    orientation: str | None = None,
) -> ImageResult:
    """Search for a public domain image.

    Args:
        query: Search term
        use_llm_selection: Use LLM to select from candidates (requires context)
        context: Article/document context for LLM selection
        orientation: "landscape", "portrait", or "square"

    Returns:
        ImageResult with URL, attribution, and metadata

    Raises:
        NoResultsError: No images found
        ImageError: Provider or configuration error
    """
    service = get_image_service()
    return await service.search(
        query,
        use_llm_selection=use_llm_selection,
        context=context,
        orientation=orientation,
    )


__all__ = [
    # Main function
    "get_image",
    # Service
    "get_image_service",
    "ImageService",
    # Types
    "ImageResult",
    "ImageSource",
    "ImageMetadata",
    "Attribution",
    # Config
    "ImageConfig",
    "get_image_config",
    # Errors
    "ImageError",
    "NoResultsError",
    "ProviderError",
    "RateLimitError",
]
