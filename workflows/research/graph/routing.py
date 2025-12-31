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

    For conduct_research, dispatches specialized researchers based on allocation:
    - Default: 1 web + 1 academic + 1 book researcher (one of each)
    - Supervisor can override via researcher_allocation field
    """
    current_status = state.get("current_status", "")

    if current_status == "conduct_research":
        pending = state.get("pending_questions", [])

        if not pending:
            logger.warning("No pending questions for research - completing")
            return "final_report"

        # Get allocation (default: 1 of each type)
        allocation = state.get("researcher_allocation") or {
            "web_count": 1,
            "academic_count": 1,
            "book_count": 1,
        }

        # All researchers use primary language config
        language_config = state.get("primary_language_config")

        sends = []
        question_idx = 0

        # Dispatch web researchers
        for _ in range(allocation.get("web_count", 1)):
            if question_idx < len(pending):
                q = pending[question_idx]
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
                question_idx += 1

        # Dispatch academic researchers
        for _ in range(allocation.get("academic_count", 1)):
            if question_idx < len(pending):
                q = pending[question_idx]
                sends.append(Send("academic_researcher", ResearcherState(
                    question=q,
                    search_queries=[],
                    search_results=[],
                    scraped_content=[],
                    thinking=None,
                    finding=None,
                    research_findings=[],
                    language_config=language_config,
                )))
                question_idx += 1

        # Dispatch book researchers
        for _ in range(allocation.get("book_count", 1)):
            if question_idx < len(pending):
                q = pending[question_idx]
                sends.append(Send("book_researcher", ResearcherState(
                    question=q,
                    search_queries=[],
                    search_results=[],
                    scraped_content=[],
                    thinking=None,
                    finding=None,
                    research_findings=[],
                    language_config=language_config,
                )))
                question_idx += 1

        if not sends:
            return "final_report"

        logger.info(
            f"Launching {len(sends)} researchers "
            f"(web={allocation.get('web_count', 1)}, "
            f"academic={allocation.get('academic_count', 1)}, "
            f"book={allocation.get('book_count', 1)})"
            + (f" language: {language_config['code']}" if language_config else "")
        )

        return sends

    elif current_status == "refine_draft":
        return "refine_draft"

    elif current_status == "research_complete":
        return "final_report"

    else:
        # Default: continue to supervisor
        return "supervisor"
