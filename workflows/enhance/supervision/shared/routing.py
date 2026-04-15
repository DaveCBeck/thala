"""
Routing functions for supervision loop graph.

These functions control the flow through the supervision subgraph,
determining when to expand on issues and when to finalize.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def route_after_analysis(state: dict[str, Any]) -> str:
    """Route based on supervisor decision.

    Called after analyze_review_node to determine next step.

    Args:
        state: Current supervision state containing the decision

    Returns:
        "expand" if research is needed, "finalize" for pass-through or error
    """
    decision = state.get("decision")
    if decision is None:
        # No decision made - treat as pass-through
        return "finalize"

    action = decision.get("action", "pass_through")

    # Handle error action - route to finalize which will trigger should_continue_supervision
    if action == "error":
        return "finalize"

    if action == "pass_through":
        return "finalize"

    return "expand"


def should_continue_supervision(state: dict[str, Any]) -> str:
    """Check if more supervision iterations are needed.

    Called after integrate_content_node to determine whether
    to loop back for another analysis or finalize.

    Args:
        state: Current supervision state with iteration tracking

    Returns:
        "continue" to loop back, "complete" to exit
    """
    # Check for failures. On any loop-level error, finalize with the
    # preserved current_review — retrying the inner node without state
    # feedback tends to be wasteful (the CLI/LLM already retried its
    # own transient failures upstream). The last-good review flows back
    # through the outer wrapper to enhance_report.
    loop_error = state.get("loop_error")
    expansion_failed = state.get("expansion_failed", False)
    integration_failed = state.get("integration_failed", False)

    if loop_error or expansion_failed or integration_failed:
        logger.warning("Loop 1 failure detected, completing with preserved state")
        return "complete"

    # Check if marked complete (pass-through was hit in a previous iteration)
    if state.get("is_complete", False):
        return "complete"

    # Check iteration limit
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if iteration >= max_iterations:
        return "complete"

    return "continue"
