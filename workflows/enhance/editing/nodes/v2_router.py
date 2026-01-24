"""V2 routing functions for the structure phase.

Provides the router node and Send-based routing for parallel rewriting.
"""

import logging
from typing import Any

from langgraph.types import Send

logger = logging.getLogger(__name__)


def v2_route_to_rewriters(state: dict) -> list[Send] | str:
    """Route to parallel section rewriters or skip to reassembly.

    If there are edit instructions, dispatch a worker for each.
    If no instructions, go directly to reassembly.

    Args:
        state: Current workflow state

    Returns:
        List of Send objects for parallel workers, or "reassemble" to skip
    """
    instructions = state.get("edit_instructions", [])

    if not instructions:
        logger.info("No edit instructions - skipping to reassembly")
        return "reassemble"

    # Build worker states for each instruction
    sends = []
    sections = state.get("sections", [])
    topic = state["input"]["topic"]
    quality_settings = state.get("quality_settings", {})

    for instr_data in instructions:
        # Build minimal worker state
        worker_state = {
            "sections": sections,
            "instruction": instr_data,
            "topic": topic,
            "quality_settings": quality_settings,
        }

        sends.append(Send("rewrite_section", worker_state))

    logger.info(f"Dispatching {len(sends)} parallel rewrite workers")
    return sends


def v2_rewrite_router_node(state: dict) -> dict[str, Any]:
    """Empty pass-through node to enable Send-based routing.

    This node exists only to provide a routing point for
    the Send() dispatch to rewrite workers.
    """
    return {}
