"""
Routing functions for supervision loop graph.

These functions control the flow through the supervision subgraph,
determining when to expand on issues and when to finalize.
"""

from typing import Any


def route_after_analysis(state: dict[str, Any]) -> str:
    """Route based on supervisor decision.

    Called after analyze_review_node to determine next step.

    Args:
        state: Current supervision state containing the decision

    Returns:
        "expand" if research is needed, "finalize" for pass-through
    """
    decision = state.get("decision")
    if decision is None:
        # No decision made - treat as pass-through
        return "finalize"

    action = decision.get("action", "pass_through")
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
    # Check if marked complete (pass-through was hit in a previous iteration)
    if state.get("is_complete", False):
        return "complete"

    # Check iteration limit
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if iteration >= max_iterations:
        return "complete"

    return "continue"
