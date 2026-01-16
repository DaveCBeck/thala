"""Graph builder for enhancement supervision workflow.

Creates a simplified orchestration graph that runs Loop 1 (theoretical depth)
and/or Loop 2 (literature expansion) on an existing report.
"""

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from workflows.enhance.supervision.nodes import (
    finalize_node,
    run_loop1_node,
    run_loop2_node,
)
from workflows.enhance.supervision.types import EnhanceState

logger = logging.getLogger(__name__)


def create_enhancement_graph(
    loops: Literal["none", "one", "two", "all"] = "all",
) -> StateGraph:
    """Create enhancement graph.

    Args:
        loops: Which loops to run:
            - "none": No loops, just pass through to finalize
            - "one": Only Loop 1 (theoretical depth)
            - "two": Only Loop 2 (literature expansion)
            - "all": Both Loop 1 and Loop 2

    Returns:
        Compiled StateGraph for enhancement
    """
    builder = StateGraph(EnhanceState)

    builder.add_node("loop1", run_loop1_node)
    builder.add_node("loop2", run_loop2_node)
    builder.add_node("finalize", finalize_node)

    if loops == "none":
        builder.add_edge(START, "finalize")
    elif loops == "one":
        builder.add_edge(START, "loop1")
        builder.add_edge("loop1", "finalize")
    elif loops == "two":
        builder.add_edge(START, "loop2")
        builder.add_edge("loop2", "finalize")
    else:  # "all"
        builder.add_edge(START, "loop1")
        builder.add_edge("loop1", "loop2")
        builder.add_edge("loop2", "finalize")

    builder.add_edge("finalize", END)

    logger.debug(f"Created enhancement graph with loops={loops}")
    return builder.compile()
