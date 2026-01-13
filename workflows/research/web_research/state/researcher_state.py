"""Researcher state and related types."""

from operator import add
from typing import Annotated, Optional
from typing_extensions import TypedDict

from .language_config import LanguageConfig


class ResearchQuestion(TypedDict):
    """A single research question for a researcher agent."""

    question_id: str
    question: str
    context: str  # Why this question matters
    priority: int  # 1=highest


class WebSearchResult(TypedDict):
    """A web search result."""

    url: str
    title: str
    description: Optional[str]
    content: Optional[str]  # Scraped content if fetched
    source_metadata: Optional[dict]  # Structured metadata for academic sources (OpenAlex)


class ResearchFinding(TypedDict):
    """Compressed finding from a researcher."""

    question_id: str
    finding: str  # Compressed research finding
    sources: list[WebSearchResult]  # Sources used
    confidence: float  # 0-1 confidence score
    gaps: list[str]  # What's still unclear
    language_code: Optional[str]  # ISO 639-1 code (e.g., "es", "zh") or None for English


class ResearcherState(TypedDict):
    """State for individual researcher agent."""

    question: ResearchQuestion
    search_queries: list[str]  # Generated search queries
    search_results: list[WebSearchResult]  # Raw search results
    scraped_content: list[str]  # Full page content
    thinking: Optional[str]  # Agent's reasoning
    finding: Optional[ResearchFinding]  # Final compressed finding
    research_findings: Annotated[list[ResearchFinding], add]  # For aggregation to parent

    # Language configuration for multi-lingual support
    language_config: Optional[LanguageConfig]  # Language this researcher operates in
