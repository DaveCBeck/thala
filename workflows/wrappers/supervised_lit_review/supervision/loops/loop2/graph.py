"""Loop 2 graph construction and state definition."""

import logging
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .analyzer import analyze_for_bases_node
from .integrator import integrate_findings_node, run_mini_review_node
from .routing import check_continue, route_after_analyze

logger = logging.getLogger(__name__)


class Loop2State(TypedDict, total=False):
    """State for Loop 2 literature base expansion.

    Uses simplified, standalone parameters - no corpus/summaries passed through.
    Papers go directly to ES/Zotero via nested academic_lit_review workflows.
    """

    # Core inputs
    current_review: str
    topic: str
    research_questions: list[str]
    quality_settings: dict

    # Iteration tracking
    iteration: int
    max_iterations: int
    explored_bases: list[str]
    is_complete: bool

    # Node outputs
    decision: dict | None

    # Error tracking
    errors: list[dict]
    iterations_failed: int
    consecutive_failures: int
    integration_failed: bool
    mini_review_failed: bool


@dataclass
class Loop2Result:
    """Result from running Loop 2."""

    current_review: str
    changes_summary: str
    explored_bases: list[str]


async def finalize_node(state: dict) -> dict:
    """Mark loop as complete and return final state."""
    logger.info("Loop 2 finalized")
    return {"is_complete": True}


def create_loop2_graph() -> StateGraph:
    """Create Loop 2 literature base expansion graph."""
    graph = StateGraph(Loop2State)

    graph.add_node("analyze_for_bases", analyze_for_bases_node)
    graph.add_node("run_mini_review", run_mini_review_node)
    graph.add_node("integrate_findings", integrate_findings_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "analyze_for_bases")
    graph.add_conditional_edges(
        "analyze_for_bases",
        route_after_analyze,
        {
            "run_mini_review": "run_mini_review",
            "finalize": "finalize",
        },
    )
    graph.add_edge("run_mini_review", "integrate_findings")
    graph.add_conditional_edges(
        "integrate_findings",
        check_continue,
        {
            "analyze_for_bases": "analyze_for_bases",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)

    return graph.compile()


loop2_graph = create_loop2_graph()
