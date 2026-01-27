"""Type definitions for Substack publishing utility."""

from typing import Literal, TypedDict


class SubstackConfig(TypedDict, total=False):
    """Configuration for Substack publishing."""

    cookies_path: str | None
    publication_url: str | None
    audience: Literal["everyone", "only_paid", "founding", "only_free"]


class CitationMapping(TypedDict):
    """Maps citation key to footnote number and content."""

    key: str  # e.g., "YFSXQJH4"
    number: int  # 1-indexed footnote number
    citation_text: str  # Full citation text for footnote


class ConversionResult(TypedDict):
    """Result of markdown to ProseMirror conversion."""

    draft_body: dict  # ProseMirror document structure
    title: str
    subtitle: str | None
    citations: list[CitationMapping]
    images_uploaded: list[str]  # S3 URLs
    warnings: list[str]


class PublishResult(TypedDict):
    """Result of publishing to Substack."""

    success: bool
    post_id: str | None
    draft_url: str | None
    publish_url: str | None
    error: str | None
