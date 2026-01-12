"""Classification types for scraped HTML pages."""

import logging
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ClassificationItem(BaseModel):
    """Classification result for a single scraped page."""

    doi: str = Field(description="DOI of the paper being classified")
    classification: Literal["full_text", "abstract_with_pdf", "paywall"] = Field(
        description="Type of content detected"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the classification (0.0-1.0)",
    )
    pdf_url: Optional[str] = Field(
        default=None,
        description="URL to PDF if classification is abstract_with_pdf. Extract from the links list.",
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen",
    )

    @field_validator("pdf_url", mode="before")
    @classmethod
    def validate_pdf_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate that pdf_url is actually a URL, not page content."""
        if v is None:
            return None
        # Quick sanity checks
        if len(v) > 2000:  # URLs shouldn't be this long
            logger.warning(f"Rejecting pdf_url: too long ({len(v)} chars)")
            return None
        if "\n" in v or "\r" in v:  # URLs don't have newlines
            logger.warning("Rejecting pdf_url: contains newlines")
            return None
        try:
            parsed = urlparse(v)
            if parsed.scheme not in ("http", "https"):
                logger.warning(f"Rejecting pdf_url: invalid scheme '{parsed.scheme}'")
                return None
            if not parsed.netloc:
                logger.warning("Rejecting pdf_url: no netloc")
                return None
            return v
        except Exception:
            logger.warning(f"Rejecting pdf_url: failed to parse")
            return None


class BatchClassificationResponse(BaseModel):
    """Response containing classifications for multiple pages."""

    items: list[ClassificationItem] = Field(
        description="List of classification results, one for each input item"
    )
