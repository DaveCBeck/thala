"""
State schemas for book finding workflow.

Defines TypedDict states for discovering books related to a theme across
three categories: analogous domain, inspiring action, and expressive fiction.
"""

from datetime import datetime
from operator import add
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from workflows.shared.language import LanguageConfig


# =============================================================================
# Reducer Functions
# =============================================================================


def merge_book_results(existing: list, new: list) -> list:
    """Merge book result lists, deduplicating by md5."""
    seen_md5 = {b.get("md5") for b in existing if b.get("md5")}
    merged = list(existing)
    for book in new:
        md5 = book.get("md5")
        if md5 and md5 not in seen_md5:
            merged.append(book)
            seen_md5.add(md5)
        elif not md5:
            merged.append(book)
    return merged


# =============================================================================
# Core Types
# =============================================================================


class BookRecommendation(TypedDict):
    """A single book recommendation from LLM."""

    title: str
    author: Optional[str]  # If suggested by LLM
    explanation: str  # 2-sentence explanation
    category: str  # "analogous" | "inspiring" | "expressive"


class BookResult(TypedDict):
    """A book found via book_search and optionally processed."""

    title: str
    authors: str
    md5: str
    url: str
    format: str
    size: str
    abstract: Optional[str]
    matched_recommendation: str  # Which recommendation title it matches
    content_summary: Optional[str]  # After processing via Marker


# =============================================================================
# Quality Settings
# =============================================================================


class BookFindingQualitySettings(TypedDict):
    """Configuration for a book finding quality tier."""

    recommendations_per_category: int
    max_concurrent_downloads: int
    max_content_for_summary: int
    recommendation_max_tokens: int
    summary_max_tokens: int
    use_opus_for_recommendations: bool


# Quality presets for book finding workflow
# Named QUALITY_PRESETS for consistency with other workflows
QUALITY_PRESETS: dict[str, BookFindingQualitySettings] = {
    "quick": BookFindingQualitySettings(
        recommendations_per_category=2,
        max_concurrent_downloads=2,
        max_content_for_summary=25000,
        recommendation_max_tokens=1024,
        summary_max_tokens=512,
        use_opus_for_recommendations=False,
    ),
    "standard": BookFindingQualitySettings(
        recommendations_per_category=3,
        max_concurrent_downloads=3,
        max_content_for_summary=50000,
        recommendation_max_tokens=2048,
        summary_max_tokens=1024,
        use_opus_for_recommendations=True,
    ),
    "comprehensive": BookFindingQualitySettings(
        recommendations_per_category=5,
        max_concurrent_downloads=5,
        max_content_for_summary=100000,
        recommendation_max_tokens=4096,
        summary_max_tokens=2048,
        use_opus_for_recommendations=True,
    ),
}

# Backwards compatibility alias
BOOK_QUALITY_PRESETS = QUALITY_PRESETS


# =============================================================================
# Input Types
# =============================================================================


class BookFindingInput(TypedDict):
    """Input parameters for book finding workflow."""

    theme: str  # The theme to explore
    brief: Optional[str]  # Optional additional context/brief
    quality: Literal["quick", "standard", "comprehensive"]
    language_code: Optional[str]  # ISO 639-1 code, default "en"


# =============================================================================
# Main State
# =============================================================================


class BookFindingState(TypedDict):
    """Complete state for book finding workflow."""

    # Input
    input: BookFindingInput
    quality_settings: BookFindingQualitySettings
    language_config: Optional[LanguageConfig]  # Full config derived from code

    # Recommendations phase (parallel aggregation via add reducer)
    analogous_recommendations: Annotated[list[BookRecommendation], add]
    inspiring_recommendations: Annotated[list[BookRecommendation], add]
    expressive_recommendations: Annotated[list[BookRecommendation], add]

    # Search phase
    search_results: Annotated[list[BookResult], merge_book_results]

    # Processing phase
    processed_books: Annotated[list[BookResult], add]
    processing_failed: Annotated[list[str], add]  # Titles that failed

    # Output
    final_markdown: Optional[str]
    final_report: Optional[str]  # Alias for final_markdown (standardized field name)

    # Workflow metadata
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    current_phase: str
    status: Optional[str]  # Standardized: "success", "partial", "failed"
    errors: Annotated[list[dict], add]
