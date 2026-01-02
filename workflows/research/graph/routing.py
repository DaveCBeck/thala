"""Routing functions for deep research graph."""

import logging
from typing import Literal

from langgraph.types import Send

from workflows.research.state import DeepResearchState, ResearcherState

logger = logging.getLogger(__name__)


def route_after_clarify(state: DeepResearchState) -> str:
    """Route based on whether clarification is needed."""
    if state.get("clarification_needed"):
        # If clarification needed but no responses yet, proceed anyway
        # (In a real implementation, this would pause for user input)
        if not state.get("clarification_responses"):
            logger.info("Clarification needed but proceeding without responses")
    return "create_brief"


def route_after_create_brief(state: DeepResearchState) -> str:
    """Route to memory search after creating brief."""
    return "search_memory"


def route_supervisor_action(state: DeepResearchState) -> str | list[Send]:
    """Route based on supervisor's chosen action.

    For conduct_research, dispatches web researchers for pending questions.
    Academic and book researchers have been removed from the main flow -
    use standalone workflows (academic_lit_review, book_finding) instead.
    """
    current_status = state.get("current_status", "")

    if current_status == "conduct_research":
        pending = state.get("pending_questions", [])

        if not pending:
            logger.warning("No pending questions for research - completing")
            return "final_report"

        # All researchers use primary language config
        language_config = state.get("primary_language_config")

        # Dispatch web researchers for each pending question (up to 3 concurrent)
        max_concurrent = 3
        sends = []
        for q in pending[:max_concurrent]:
            sends.append(Send("web_researcher", ResearcherState(
                question=q,
                search_queries=[],
                search_results=[],
                scraped_content=[],
                thinking=None,
                finding=None,
                research_findings=[],
                language_config=language_config,
            )))

        if not sends:
            return "final_report"

        logger.info(
            f"Launching {len(sends)} web researchers"
            + (f" (language: {language_config['code']})" if language_config else "")
        )

        return sends

    elif current_status == "refine_draft":
        return "refine_draft"

    elif current_status == "research_complete":
        return "final_report"

    else:
        # Default: continue to supervisor
        return "supervisor"
