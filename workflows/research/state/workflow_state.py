"""Main workflow state for deep research."""

from datetime import datetime
from operator import add
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .input_types import ClarificationQuestion, ResearchBrief, ResearchInput
from .language_config import LanguageConfig, TranslationConfig
from .researcher_state import ResearchFinding, ResearchQuestion
from .researcher_types import ResearcherAllocation
from .supervisor_state import DiffusionState, DraftReport


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
