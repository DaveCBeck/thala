"""Data models for extraction results."""

from pydantic import BaseModel, Field


class PaperSummarySchema(BaseModel):
    """Schema for full-text paper summary extraction."""

    key_findings: list[str] = Field(
        default_factory=list,
        description="3-5 specific findings from the paper",
    )
    methodology: str = Field(
        default="Not specified",
        description="Brief research method description (1-2 sentences)",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Stated limitations from the paper",
    )
    future_work: list[str] = Field(
        default_factory=list,
        description="Suggested future research directions",
    )
    themes: list[str] = Field(
        default_factory=list,
        description="3-5 topic tags for clustering",
    )
