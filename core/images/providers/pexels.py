"""Pexels image provider.

API: https://www.pexels.com/api/documentation/
Rate limit: 200 requests/hour (free tier)
License: Pexels License (free for commercial use, no attribution required)
"""

import logging

from core.utils.http_errors import safe_http_request

from ..errors import ProviderError, RateLimitError
from ..types import Attribution, ImageMetadata, ImageResult, ImageSource
from .base import BaseImageProvider

logger = logging.getLogger(__name__)

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
            response = await safe_http_request(
                client,
                "GET",
                f"{BASE_URL}/search",
                error_class=ProviderError,
                params=params,
                headers={"Authorization": self._config.pexels_api_key},
            )
        except ProviderError as e:
            if "429" in str(e):
                raise RateLimitError("Pexels rate limit exceeded", provider="pexels") from e
            raise

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
