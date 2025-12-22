"""
State schemas for deep research workflow.

Defines TypedDict states for:
- Main workflow (DeepResearchState)
- Supervisor diffusion algorithm (DiffusionState)
- Individual researcher agents (ResearcherState)
"""

from datetime import datetime
from operator import add
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# =============================================================================
# Input Types
# =============================================================================


class ResearchInput(TypedDict):
    """Initial user research request."""

    query: str  # Original user query
    depth: Literal["quick", "standard", "comprehensive"]
    max_sources: int  # Max web sources to use
    max_iterations: Optional[int]  # Override default for depth


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


class ResearcherState(TypedDict):
    """State for individual researcher agent."""

    question: ResearchQuestion
    search_queries: list[str]  # Generated search queries
    search_results: list[WebSearchResult]  # Raw search results
    scraped_content: list[str]  # Full page content
    thinking: Optional[str]  # Agent's reasoning
    finding: Optional[ResearchFinding]  # Final compressed finding
    research_findings: Annotated[list[ResearchFinding], add]  # For aggregation to parent


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
