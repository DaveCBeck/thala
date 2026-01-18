"""Supervision loop subgraph for iterative review improvement (Loop 1).

Implements an iterative loop that:
1. Analyzes the review for theoretical gaps
2. If gap found: expands on the topic, integrates findings
3. Loops until max iterations or pass-through

This is a standalone copy for the enhancement workflow, importing nodes
and routing from the original supervised_lit_review location.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from workflows.shared.tracing import workflow_traceable, merge_trace_config
from workflows.shared.workflow_state_store import save_workflow_state
from workflows.enhance.supervision.shared.nodes import (
    analyze_review_node,
    expand_topic_node,
    integrate_content_node,
)
from workflows.enhance.supervision.shared.routing import (
    route_after_analysis,
    should_continue_supervision,
)

logger = logging.getLogger(__name__)


class Loop1State(TypedDict, total=False):
    """State schema for Loop 1 (theoretical depth supervision).

    Uses simplified, standalone parameters - no nested dicts or corpus objects.
    """

    # Core inputs
    current_review: str
    topic: str
    research_questions: list[str]
    source_count: int

    # Quality config
    quality_settings: dict[str, Any]
    max_iterations: int

    # Iteration tracking
    iteration: int
    issues_explored: list[str]
    is_complete: bool

    # Node outputs
    decision: Optional[dict]
    expansion_result: Optional[dict]
    supervision_expansions: list[dict]

    # Final outputs
    final_review: Optional[str]
    completion_reason: Optional[str]

    # Papers added during this loop
    paper_corpus: dict[str, Any]
    paper_summaries: dict[str, Any]
    zotero_keys: dict[str, str]

    # Error tracking
    loop_error: Optional[dict]
    expansion_failed: bool
    integration_failed: bool
    consecutive_failures: int


@dataclass
class Loop1Result:
    """Result from running Loop 1."""

    current_review: str
    changes_summary: str
    issues_explored: list[str]


def finalize_loop1_node(state: dict[str, Any]) -> dict[str, Any]:
    """Finalize Loop 1 and prepare output."""
    current_review = state.get("current_review", "")
    iteration = state.get("iteration", 0)
    is_complete = state.get("is_complete", False)
    decision = state.get("decision", {})

    if is_complete or decision.get("action") == "pass_through":
        completion_reason = "Supervisor approved theoretical depth"
    else:
        max_iterations = state.get("max_iterations", 3)
        completion_reason = f"Reached maximum iterations ({max_iterations})"

    logger.info(
        f"Finalizing Loop 1 after {iteration} iterations. Reason: {completion_reason}"
    )

    return {
        "final_review": current_review,
        "is_complete": True,
        "completion_reason": completion_reason,
    }


def create_loop1_graph() -> StateGraph:
    """Create the Loop 1 subgraph.

    Flow:
        START -> analyze_review -> route
            -> (research_needed) -> expand_topic -> integrate_content
                -> check_continue -> (continue) -> analyze_review
                               -> (complete) -> finalize -> END
            -> (pass_through) -> finalize -> END
    """
    builder = StateGraph(Loop1State)

    builder.add_node("analyze_review", analyze_review_node)
    builder.add_node("expand_topic", expand_topic_node)
    builder.add_node("integrate_content", integrate_content_node)
    builder.add_node("finalize", finalize_loop1_node)

    builder.add_edge(START, "analyze_review")

    builder.add_conditional_edges(
        "analyze_review",
        route_after_analysis,
        {
            "expand": "expand_topic",
            "finalize": "finalize",
        },
    )

    builder.add_edge("expand_topic", "integrate_content")

    builder.add_conditional_edges(
        "integrate_content",
        should_continue_supervision,
        {
            "continue": "analyze_review",
            "complete": "finalize",
        },
    )

    builder.add_edge("finalize", END)

    return builder.compile()


loop1_graph = create_loop1_graph()


@workflow_traceable(name="Loop1_TheoreticalDepth", workflow_type="supervision_loop1")
async def run_loop1_standalone(
    review: str,
    topic: str,
    research_questions: list[str],
    max_iterations: int = 3,
    source_count: int = 0,
    quality_settings: dict[str, Any] | None = None,
    config: dict | None = None,
) -> Loop1Result:
    """Run Loop 1 (theoretical depth supervision) as a standalone operation.

    Args:
        review: The literature review text to analyze and improve
        topic: Research topic
        research_questions: List of research questions
        max_iterations: Maximum supervision iterations
        source_count: Number of sources currently in corpus (for context)
        quality_settings: Quality settings for focused expansion
        config: Optional LangGraph config for tracing

    Returns:
        Loop1Result with improved review and metadata
    """
    initial_state: Loop1State = {
        "current_review": review,
        "topic": topic,
        "research_questions": research_questions,
        "source_count": source_count,
        "quality_settings": quality_settings or {},
        "max_iterations": max_iterations,
        "iteration": 0,
        "issues_explored": [],
        "is_complete": False,
        "decision": None,
        "expansion_result": None,
        "supervision_expansions": [],
        "final_review": None,
        "completion_reason": None,
        "paper_corpus": {},
        "paper_summaries": {},
        "zotero_keys": {},
        "loop_error": None,
        "expansion_failed": False,
        "integration_failed": False,
        "consecutive_failures": 0,
    }

    logger.info(
        f"Starting Loop 1: max_iterations={max_iterations}, "
        f"review length={len(review)} chars"
    )

    final_state = await loop1_graph.ainvoke(
        initial_state, config=merge_trace_config(config)
    )

    expansions = final_state.get("supervision_expansions", [])
    issues_explored = final_state.get("issues_explored", [])

    # Build changes summary
    if expansions:
        topics_explored = [exp.get("topic", "unknown") for exp in expansions]
        changes_summary = (
            f"Explored {len(expansions)} theoretical gaps: {', '.join(topics_explored)}"
        )
    else:
        changes_summary = "No theoretical gaps identified"

    logger.info(f"Loop 1 complete: {len(issues_explored)} issues explored")

    # Save state for analysis (dev mode only)
    run_id = config.get("run_id", "unknown") if config else "unknown"
    save_workflow_state(
        workflow_name="enhance_supervision_loop1",
        run_id=str(run_id),
        state={
            "input": {
                "review_length": len(review),
                "topic": topic,
                "research_questions": research_questions,
                "max_iterations": max_iterations,
                "source_count": source_count,
            },
            "output": {
                "review_length": len(final_state.get("final_review", review)),
                "issues_explored": issues_explored,
                "expansions": expansions,
            },
            "final_state": final_state,
        },
    )

    return Loop1Result(
        current_review=final_state.get("final_review", review),
        changes_summary=changes_summary,
        issues_explored=issues_explored,
    )
