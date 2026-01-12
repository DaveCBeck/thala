"""Graph construction for supervision orchestration."""

from langgraph.graph import END, START, StateGraph

from .types import OrchestrationState
from .nodes import (
    run_loop1_node,
    run_loop2_node,
    run_loop3_node,
    run_loop4_node,
    run_loop4_5_node,
    run_loop5_node,
    finalize_node,
)
from .routing import increment_loop3_repeat_node, route_after_loop4_5


def create_orchestration_graph() -> StateGraph:
    """Create the multi-loop orchestration graph."""
    builder = StateGraph(OrchestrationState)

    builder.add_node("loop1", run_loop1_node)
    builder.add_node("loop2", run_loop2_node)
    builder.add_node("loop3", run_loop3_node)
    builder.add_node("loop4", run_loop4_node)
    builder.add_node("loop4_5", run_loop4_5_node)
    builder.add_node("increment_and_loop3", increment_loop3_repeat_node)
    builder.add_node("loop5", run_loop5_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "loop1")
    builder.add_edge("loop1", "loop2")
    builder.add_edge("loop2", "loop3")
    builder.add_edge("loop3", "loop4")
    builder.add_edge("loop4", "loop4_5")

    builder.add_conditional_edges(
        "loop4_5",
        route_after_loop4_5,
        {
            "increment_and_loop3": "increment_and_loop3",
            "loop5": "loop5",
        },
    )

    builder.add_edge("increment_and_loop3", "loop3")

    builder.add_edge("loop5", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
