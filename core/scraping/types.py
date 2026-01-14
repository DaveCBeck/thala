"""Types for unified URL content retrieval.

This module defines the result types and configuration options for the
unified get_url() function that handles all URL types.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ContentSource(str, Enum):
    """How the content was acquired."""

    SCRAPED = "scraped"  # Web scraping (Firecrawl/Playwright)
    PDF_DIRECT = "pdf_direct"  # Direct PDF download → Marker
    PDF_EXTRACTED = "pdf_extracted"  # PDF link extracted from abstract page
    RETRIEVE_ACADEMIC = "retrieve_academic"  # retrieve-academic service


class ContentClassification(str, Enum):
    """Classification of academic content."""

    FULL_TEXT = "full_text"  # Complete article ready for processing
    ABSTRACT_WITH_PDF = "abstract_with_pdf"  # Abstract page with PDF download link
    PAYWALL = "paywall"  # Access restricted, needs fallback
    NON_ACADEMIC = "non_academic"  # Not academic content


class GetUrlResult(BaseModel):
    """Result from get_url() function."""

    # Original input
    url: str  # Original URL/DOI requested
    resolved_url: Optional[str] = None  # Final URL after resolution/redirects

    # Content
    content: str  # Markdown content
    content_type: str = "markdown"  # Always markdown for now

    # Metadata
    source: ContentSource  # How content was acquired
    provider: str  # e.g., "firecrawl-local", "marker", "retrieve-academic"

    # Academic metadata (when applicable)
    doi: Optional[str] = None  # DOI if detected/resolved
    classification: Optional[ContentClassification] = None

    # Links extracted (for further processing)
    links: list[str] = Field(default_factory=list)

    # Debugging
    fallback_chain: list[str] = Field(default_factory=list)  # Attempted sources


class GetUrlOptions(BaseModel):
    """Options for get_url() call."""

    # PDF processing
    pdf_quality: str = "balanced"  # fast, balanced, quality
    pdf_langs: list[str] = Field(default_factory=lambda: ["English"])

    # Academic detection
    detect_academic: bool = True  # Run content classification
    allow_retrieve_academic: bool = True  # Allow retrieve-academic fallback

    # Scraping options
    include_links: bool = True  # Extract links from pages

    # Timeouts (seconds)
    scrape_timeout: float = 60.0
    pdf_timeout: Optional[float] = (
        None  # None = no client-side limit, uses Marker queue limits
    )
    retrieve_academic_timeout: float = 180.0


class DoiInfo(BaseModel):
    """Information about a detected DOI."""

    doi: str  # Normalized DOI (e.g., "10.1234/example")
    doi_url: str  # Full URL form (https://doi.org/...)
    source: str  # Where DOI was found: "input", "url", "content"
