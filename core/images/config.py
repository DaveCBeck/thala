"""Configuration for image aggregator."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ImageConfig:
    """Configuration for image search service.

    Environment Variables:
        PEXELS_API_KEY: API key for Pexels (required for Pexels provider)
        UNSPLASH_API_KEY: API key for Unsplash (required for Unsplash provider)
        IMAGE_CACHE_TTL_DAYS: Cache TTL in days (default: 30)
        IMAGE_TIMEOUT: Request timeout in seconds (default: 15)
    """

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


_config: ImageConfig | None = None


def get_image_config() -> ImageConfig:
    """Get global ImageConfig instance."""
    global _config
    if _config is None:
        _config = ImageConfig()
    return _config
