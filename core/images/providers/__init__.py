"""Image provider registry."""

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
