"""Base provider class for image sources."""

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
