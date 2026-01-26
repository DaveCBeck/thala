"""Type definitions for image aggregator."""

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
    """Result from image search."""

    url: str = Field(description="Full-size image URL")
    thumbnail_url: str | None = None
    attribution: Attribution
    metadata: ImageMetadata
    provider: ImageSource
    query: str = Field(description="Original search term")
