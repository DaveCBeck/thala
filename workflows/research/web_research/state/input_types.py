"""Input types for research workflow."""

from typing import Optional
from typing_extensions import TypedDict

from core.llm_broker import UserMode
from workflows.shared.quality_config import QualityTier


class RecencyFilter(TypedDict):
    """Date filtering for recency-aware web search.

    Only a fraction (quota) of search iterations use date filtering,
    preserving full coverage while guaranteeing recent signal.
    """

    after_date: str  # ISO YYYY-MM-DD — "search only after this date"
    quota: float  # 0.0-1.0 — fraction of search calls that use date filtering


class ResearchInput(TypedDict):
    """Initial user research request."""

    query: str  # Original user query
    quality: QualityTier
    max_iterations: Optional[int]  # Override default for quality
    language: Optional[str]  # ISO 639-1 code (e.g., "es", "zh") - default is English
    llm_mode: Optional[UserMode]  # LLM broker mode (Fast/Balanced/Economical)
    recency_filter: Optional[RecencyFilter]  # Date filtering for recent signal


class ClarificationQuestion(TypedDict):
    """Question to clarify user intent."""

    question: str
    options: Optional[list[str]]  # Suggested answers (if applicable)


class ResearchBrief(TypedDict):
    """Refined research brief after clarification."""

    topic: str  # Core research topic
    objectives: list[str]  # Specific research objectives
    scope: str  # What's in/out of scope
    key_questions: list[str]  # Questions to answer
    memory_context: str  # Relevant memory findings summary
