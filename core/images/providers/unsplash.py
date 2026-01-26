"""Unsplash image provider.

API: https://unsplash.com/documentation
Rate limit: 50 requests/hour (demo), 5000/hour (production)
License: Unsplash License (free, attribution required)
"""

import logging

from core.utils.http_errors import safe_http_request

from ..errors import ProviderError, RateLimitError
from ..types import Attribution, ImageMetadata, ImageResult, ImageSource
from .base import BaseImageProvider

logger = logging.getLogger(__name__)

BASE_URL = "https://api.unsplash.com"


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
            raise ProviderError(
                "Unsplash API key not configured", provider="unsplash"
            )

        client = await self._get_client()

        params: dict = {"query": query, "per_page": limit}
        if orientation:
            params["orientation"] = orientation

        try:
            response = await safe_http_request(
                client,
                "GET",
                f"{BASE_URL}/search/photos",
                error_class=ProviderError,
                params=params,
                headers={
                    "Authorization": f"Client-ID {self._config.unsplash_api_key}"
                },
            )
        except ProviderError as e:
            if "429" in str(e) or "403" in str(e):
                raise RateLimitError(
                    "Unsplash rate limit exceeded", provider="unsplash"
                ) from e
            raise

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
