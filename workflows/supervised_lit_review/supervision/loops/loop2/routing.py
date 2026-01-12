"""Loop 2 routing logic."""

import logging

logger = logging.getLogger(__name__)


def route_after_analyze(state: dict) -> str:
    """Route based on analyzer decision."""
    decision = state.get("decision")
    if not decision:
        return "finalize"

    if decision["action"] == "error":
        return "finalize"

    if decision["action"] == "expand_base":
        return "run_mini_review"
    return "finalize"


def check_continue(state: dict) -> str:
    """Check if should continue iterating or complete."""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    integration_failed = state.get("integration_failed", False)
    mini_review_failed = state.get("mini_review_failed", False)
    consecutive_failures = state.get("consecutive_failures", 0)

    if integration_failed or mini_review_failed:
        if consecutive_failures >= 2:
            logger.warning("Too many consecutive failures, completing Loop 2")
            return "finalize"
        return "analyze_for_bases"

    if iteration >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached")
        return "finalize"

    return "analyze_for_bases"
