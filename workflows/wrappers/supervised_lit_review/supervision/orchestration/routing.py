"""Routing functions for supervision orchestration."""

import logging
from typing import Any

from .types import OrchestrationState

logger = logging.getLogger(__name__)


def increment_loop3_repeat_node(state: OrchestrationState) -> dict[str, Any]:
    """Increment loop3_repeat_count when returning from Loop 4.5 to Loop 3."""
    current_count = state.get("loop3_repeat_count", 0)
    logger.info(f"Incrementing loop3_repeat_count from {current_count} to {current_count + 1}")
    return {"loop3_repeat_count": current_count + 1}


def route_after_loop4_5(state: OrchestrationState) -> str:
    """Route after Loop 4.5 cohesion check."""
    loop4_5_result = state.get("loop4_5_result", {})
    needs_restructuring = loop4_5_result.get("needs_restructuring", False)
    repeat_count = state.get("loop3_repeat_count", 0)

    max_repeats = state["loop_progress"]["max_iterations_per_loop"]
    if needs_restructuring and repeat_count < max_repeats:
        logger.info("Cohesion check: returning to Loop 3 for restructuring")
        return "increment_and_loop3"
    else:
        if needs_restructuring:
            logger.warning(
                "Cohesion check flagged restructuring but max repeats reached, "
                "proceeding to Loop 5"
            )
        else:
            logger.info("Cohesion check passed, proceeding to Loop 5")
        return "loop5"
