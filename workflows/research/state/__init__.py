"""
State schemas for deep research workflow.

Defines TypedDict states for:
- Main workflow (DeepResearchState)
- Supervisor diffusion algorithm (DiffusionState)
- Individual researcher agents (ResearcherState)
- Language configuration for multi-lingual support
"""

from .researcher_types import ResearcherAllocation, parse_allocation
from .language_config import LanguageConfig, TranslationConfig
from .input_types import ResearchInput, ClarificationQuestion, ResearchBrief
from .researcher_state import ResearchQuestion, WebSearchResult, ResearchFinding, ResearcherState
from .supervisor_state import DraftReport, DiffusionState, calculate_completeness
from .supervisor_tools import ConductResearch, ResearchComplete, RefineDraftReport, SupervisorDecision
from .researcher_tools import SearchQueries, QueryValidation, QueryValidationBatch
from .workflow_state import DeepResearchState

__all__ = [
    # Researcher types
    "ResearcherAllocation", "parse_allocation",
    # Language config
    "LanguageConfig", "TranslationConfig",
    # Input types
    "ResearchInput", "ClarificationQuestion", "ResearchBrief",
    # Researcher state
    "ResearchQuestion", "WebSearchResult", "ResearchFinding", "ResearcherState",
    # Supervisor state
    "DraftReport", "DiffusionState", "calculate_completeness",
    # Supervisor tools
    "ConductResearch", "ResearchComplete", "RefineDraftReport", "SupervisorDecision",
    # Researcher tools
    "SearchQueries", "QueryValidation", "QueryValidationBatch",
    # Workflow state
    "DeepResearchState",
]
