"""Types for content classification."""

from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class ClassificationResult(BaseModel):
    """Result of content classification."""

    classification: Literal["full_text", "abstract_with_pdf", "paywall", "non_academic"]
    confidence: float = Field(ge=0.0, le=1.0)
    pdf_url: Optional[str] = None
    reasoning: str
    # Metadata for DOI lookup when paywall detected
    title: Optional[str] = None
    authors: Optional[list[str]] = None

    @field_validator("pdf_url", mode="before")
    @classmethod
    def validate_pdf_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate PDF URL is actually a URL, not page content."""
        if v is None:
            return None
        if not isinstance(v, str):
            return None
        # Check length and basic structure
        if len(v) > 2000 or "\n" in v:
            return None
        try:
            parsed = urlparse(v)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                return None
            return v
        except Exception:
            return None
