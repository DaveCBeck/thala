"""
State schemas for deep research workflow.

Defines TypedDict states for:
- Main workflow (DeepResearchState)
- Supervisor diffusion algorithm (DiffusionState)
- Individual researcher agents (ResearcherState)
- Language configuration for multi-lingual support
"""

from datetime import datetime
from enum import Enum
from operator import add
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# =============================================================================
# Researcher Types
# =============================================================================


class ResearcherType(str, Enum):
    """Types of specialized researchers."""

    WEB = "web"           # Firecrawl + Perplexity
    ACADEMIC = "academic" # OpenAlex
    BOOK = "book"         # book_search


class ResearcherAllocation(TypedDict):
    """Allocation of researchers for a conduct_research action."""

    web_count: int       # Number of web researchers (default: 1)
    academic_count: int  # Number of academic researchers (default: 1)
    book_count: int      # Number of book researchers (default: 1)


def parse_allocation(allocation_str: str) -> ResearcherAllocation:
    """Parse allocation string like '111', '210', '300' into ResearcherAllocation.

    Format: 3-character string where each digit represents count of:
    - Position 0: web researchers
    - Position 1: academic researchers
    - Position 2: book researchers

    Total must not exceed 3 (MAX_CONCURRENT_RESEARCHERS).

    Args:
        allocation_str: String like "111" (1 web, 1 academic, 1 book)
                       or "300" (3 web, 0 academic, 0 book)

    Returns:
        ResearcherAllocation dict

    Raises:
        ValueError: If string is invalid format or total exceeds 3

    Examples:
        >>> parse_allocation("111")
        {'web_count': 1, 'academic_count': 1, 'book_count': 1}
        >>> parse_allocation("210")
        {'web_count': 2, 'academic_count': 1, 'book_count': 0}
        >>> parse_allocation("300")
        {'web_count': 3, 'academic_count': 0, 'book_count': 0}
    """
    if not allocation_str or len(allocation_str) != 3:
        raise ValueError(
            f"Allocation must be a 3-character string (e.g., '111', '210'), got: {allocation_str!r}"
        )

    try:
        web = int(allocation_str[0])
        academic = int(allocation_str[1])
        book = int(allocation_str[2])
    except ValueError:
        raise ValueError(
            f"Allocation must contain only digits (0-3), got: {allocation_str!r}"
        )

    if not all(0 <= x <= 3 for x in [web, academic, book]):
        raise ValueError(
            f"Each allocation digit must be 0-3, got: {allocation_str!r}"
        )

    total = web + academic + book
    if total > 3:
        raise ValueError(
            f"Total allocation must not exceed 3, got {total} from '{allocation_str}'"
        )

    if total == 0:
        raise ValueError(
            f"Total allocation must be at least 1, got 0 from '{allocation_str}'"
        )

    return ResearcherAllocation(
        web_count=web,
        academic_count=academic,
        book_count=book,
    )


# =============================================================================
# Language Configuration Types
# =============================================================================


class LanguageConfig(TypedDict):
    """Configuration for a specific language in the research workflow."""

    code: str  # ISO 639-1 code (e.g., "es", "zh", "ja")
    name: str  # Full language name (e.g., "Spanish", "Mandarin Chinese")
    search_domains: list[str]  # Preferred domain TLDs (e.g., [".es", ".mx"])
    search_engine_locale: str  # Locale code for search APIs (e.g., "es-ES")


class TranslationConfig(TypedDict):
    """Configuration for translating the final research output."""

    enabled: bool  # Whether to translate the final report
    target_language: str  # Target language code (e.g., "en")
    preserve_quotes: bool  # Keep direct quotes in original language
    preserve_citations: bool  # Keep citation format unchanged


# =============================================================================
# Input Types
# =============================================================================


class ResearchInput(TypedDict):
    """Initial user research request."""

    query: str  # Original user query
    depth: Literal["quick", "standard", "comprehensive"]
    max_sources: int  # Max web sources to use
    max_iterations: Optional[int]  # Override default for depth

    # Language configuration
    language: Optional[str]  # Single language mode: ISO 639-1 code (e.g., "es", "zh")
    translate_to: Optional[str]  # Translate final output to this language
    preserve_quotes: Optional[bool]  # Keep quotes in original language when translating


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


# =============================================================================
# Researcher Types
# =============================================================================


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


# =============================================================================
# Supervisor Types
# =============================================================================


class DraftReport(TypedDict):
    """Draft report being refined."""

    content: str
    version: int
    last_updated: datetime
    gaps_remaining: list[str]  # Research gaps to fill


class DiffusionState(TypedDict):
    """State for diffusion algorithm."""

    iteration: int  # Current iteration
    max_iterations: int  # Max before forced completion
    completeness_score: float  # 0-1 estimated completeness
    areas_explored: list[str]  # Topics already researched
    areas_to_explore: list[str]  # Topics still needed
    last_decision: str  # Last decision made by supervisor


def calculate_completeness(
    findings: list["ResearchFinding"],
    key_questions: list[str],
    iteration: int,
    max_iterations: int,
    gaps_remaining: list[str] | None = None,
) -> float:
    """Calculate research completeness from multiple signals.

    Uses a weighted multi-signal formula:
    - 40%: Iteration progress (gives baseline progression)
    - 30%: Findings coverage (questions answered with good confidence)
    - 20%: Average confidence of findings
    - 15%: Gap penalty (reduces score based on known gaps)

    This ensures:
    - Score increases during research phase (not stuck at 0%)
    - Score reflects quality (confidence)
    - Natural progression toward 85% threshold for completion

    Args:
        findings: Research findings collected so far
        key_questions: Initial research questions from brief
        iteration: Current iteration number
        max_iterations: Maximum iterations for this depth
        gaps_remaining: Known gaps from draft refinement (optional)

    Returns:
        Completeness score between 0.0 and 1.0
    """
    gaps_remaining = gaps_remaining or []

    # 1. Iteration progress (40% weight) - capped at 90% contribution
    iteration_score = min(iteration / max(max_iterations, 1), 0.9)

    # 2. Findings coverage (30% weight)
    total_questions = max(len(key_questions), 1)
    high_confidence_findings = sum(
        1 for f in findings if f.get("confidence", 0) > 0.5
    )
    coverage_score = min(high_confidence_findings / total_questions, 1.0)

    # 3. Average confidence (20% weight)
    if findings:
        avg_confidence = sum(f.get("confidence", 0.5) for f in findings) / len(findings)
    else:
        avg_confidence = 0.0

    # 4. Gap penalty (15% weight, inverted) - less punishing, capped at 10 gaps
    gap_score = max(0, 1.0 - min(len(gaps_remaining), 10) * 0.05)

    # Weighted sum
    completeness = (
        0.40 * iteration_score
        + 0.30 * coverage_score
        + 0.20 * avg_confidence
        + 0.15 * gap_score
    )

    return min(completeness, 1.0)


# =============================================================================
# Supervisor Tool Schemas (Pydantic for tool binding)
# =============================================================================


class ConductResearch(BaseModel):
    """Tool for delegating a research task to a specialized sub-agent."""

    research_topic: str = Field(
        description="The topic to research. Should be a single topic described in high detail (at least a paragraph)."
    )


class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""

    pass


class RefineDraftReport(BaseModel):
    """Tool for refining the draft report with new findings."""

    updates: str = Field(
        description="The updates to make to the draft report based on new findings."
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Remaining gaps that still need research.",
    )


class SupervisorDecision(BaseModel):
    """Supervisor's structured decision for the next research step.

    Used with structured output to ensure clean, parseable decisions
    without metadata contamination.
    """

    action: Literal["conduct_research", "refine_draft", "research_complete"] = Field(
        description="The next action to take in the research process."
    )

    # For conduct_research action
    research_questions: list[str] = Field(
        default_factory=list,
        description="1-3 specific research questions to investigate. Must be actual questions about the research topic - NOT analysis notes, metadata, or summaries of previous findings.",
        max_length=3,
    )

    # Researcher allocation (for conduct_research action)
    web_researchers: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Number of web researchers (0-3). Best for current events, tech trends, tools, products.",
    )
    academic_researchers: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Number of academic researchers (0-3). Best for peer-reviewed research, clinical studies, scientific methodology.",
    )
    book_researchers: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Number of book researchers (0-3). Best for foundational theory, historical context, comprehensive overviews.",
    )
    allocation_reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of why this researcher allocation was chosen based on topic type.",
    )

    # For refine_draft action
    draft_updates: Optional[str] = Field(
        default=None,
        description="Content to add or update in the draft report based on new findings.",
    )
    remaining_gaps: list[str] = Field(
        default_factory=list,
        description="Research gaps that still need investigation.",
    )

    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why this action was chosen.",
    )


# =============================================================================
# Researcher Structured Output Schemas
# =============================================================================


class SearchQueries(BaseModel):
    """Generated search queries for a research question."""

    queries: list[str] = Field(
        description="2-3 specific search queries to find authoritative sources. Each query should be targeted and focus only on the research topic.",
        min_length=1,
        max_length=5,
    )


class QueryValidation(BaseModel):
    """Validation result for a single search query."""

    is_relevant: bool = Field(
        description="Whether the query is relevant to the research question"
    )
    reason: str = Field(
        description="Brief explanation of why the query is or isn't relevant"
    )


class QueryValidationBatch(BaseModel):
    """Batch validation of search queries."""

    validations: list[QueryValidation] = Field(
        description="Validation results for each query in order"
    )


# =============================================================================
# Main Workflow State
# =============================================================================


class DeepResearchState(TypedDict):
    """Main workflow state for deep research."""

    # Input
    input: ResearchInput

    # Clarification phase
    clarification_needed: bool
    clarification_questions: list[ClarificationQuestion]
    clarification_responses: Optional[dict[str, str]]

    # Brief creation
    research_brief: Optional[ResearchBrief]

    # Memory search results
    memory_findings: list[dict]  # Results from search_memory
    memory_context: str  # Summarized memory context

    # Iterate plan - customized research approach
    research_plan: Optional[str]  # Customized plan based on memory

    # Research phase (parallel writes via Annotated[..., add])
    pending_questions: list[ResearchQuestion]
    active_researchers: int  # Currently running (max 3)
    research_findings: Annotated[list[ResearchFinding], add]

    # Researcher allocation for specialized researchers
    researcher_allocation: Optional[ResearcherAllocation]  # {web_count, academic_count, book_count}

    # Supervisor messages (for tool-based agent)
    supervisor_messages: Annotated[list[BaseMessage], add_messages]

    # Diffusion algorithm
    diffusion: DiffusionState

    # Draft refinement
    draft_report: Optional[DraftReport]

    # Final output
    final_report: Optional[str]
    citations: list[dict]  # Structured citations
    citation_keys: list[str]  # Zotero keys created for citations

    # Store integration
    store_record_id: Optional[str]  # UUID of saved research
    zotero_key: Optional[str]  # If saved to Zotero

    # Error tracking
    errors: Annotated[list[dict], add]

    # Workflow metadata
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    current_status: str
    langsmith_run_id: Optional[str]  # LangSmith trace ID for run inspection

    # ==========================================================================
    # Language support
    # ==========================================================================

    # Primary language for single-language mode
    primary_language: Optional[str]  # ISO 639-1 code (default: "en")
    primary_language_config: Optional[LanguageConfig]  # Full config for primary language

    # Translation
    translation_config: Optional[TranslationConfig]  # If translating final output
    translated_report: Optional[str]  # Translated version of final report
