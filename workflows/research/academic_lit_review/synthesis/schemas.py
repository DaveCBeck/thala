"""Pydantic models for structured synthesis outputs."""

from typing import Literal

from pydantic import BaseModel, Field


class QualityCheckOutput(BaseModel):
    """Pydantic model for quality check output."""

    issues: list[str] = Field(default_factory=list, description="Quality issues found")
    suggestions: list[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    overall_quality: Literal["good", "acceptable", "needs_revision"] = Field(
        description="Overall quality assessment"
    )
    citation_issues: list[str] = Field(
        default_factory=list, description="Specific citation problems"
    )
