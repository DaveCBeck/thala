"""Supervision loop subgraph for iterative review improvement.

Implements an iterative loop that:
1. Analyzes the review for theoretical gaps
2. If gap found: expands on the topic, integrates findings
3. Loops until max iterations or pass-through
"""

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from workflows.academic_lit_review.state import (
    LitReviewInput,
    PaperMetadata,
    PaperSummary,
    QualitySettings,
    ThematicCluster,
)
from workflows.supervised_lit_review.supervision.nodes import (
    analyze_review_node,
    expand_topic_node,
    integrate_content_node,
)
from workflows.supervised_lit_review.supervision.routing import (
    route_after_analysis,
    should_continue_supervision,
)

logger = logging.getLogger(__name__)


class SupervisionSubgraphState(TypedDict, total=False):
    """State schema for the supervision subgraph.

    Defines all fields to ensure LangGraph properly preserves state
    across node transitions. Uses total=False to allow partial updates.
    """

    # Review content
    current_review: str
    final_review_v2: Optional[str]

    # Context from main workflow
    input: LitReviewInput
    paper_corpus: dict[str, PaperMetadata]
    paper_summaries: dict[str, PaperSummary]
    clusters: list[ThematicCluster]
    quality_settings: QualitySettings
    zotero_keys: dict[str, str]

    # Supervision tracking
    iteration: int
    max_iterations: int
    supervision_depth: int
    issues_explored: list[str]
    is_complete: bool

    # Outputs
    supervision_expansions: list[dict]
    decision: Optional[dict]
    expansion_result: Optional[dict]
    completion_reason: Optional[str]


def finalize_supervision_node(state: dict[str, Any]) -> dict[str, Any]:
    """Finalize the supervision loop and prepare output.

    Sets the final review and marks supervision as complete.
    """
    current_review = state.get("current_review", "")
    iteration = state.get("iteration", 0)
    is_complete = state.get("is_complete", False)
    decision = state.get("decision", {})

    # Determine reason for completion
    if is_complete or decision.get("action") == "pass_through":
        completion_reason = "Supervisor approved theoretical depth"
    else:
        max_iterations = state.get("max_iterations", 3)
        completion_reason = f"Reached maximum iterations ({max_iterations})"

    logger.info(
        f"Finalizing supervision after {iteration} iterations. "
        f"Reason: {completion_reason}"
    )

    return {
        "final_review_v2": current_review,
        "is_complete": True,
        "completion_reason": completion_reason,
    }


def create_supervision_subgraph() -> StateGraph:
    """Create the supervision loop subgraph.

    Flow:
        START -> analyze_review -> route
            -> (research_needed) -> expand_topic -> integrate_content
                -> check_continue -> (continue) -> analyze_review
                               -> (complete) -> finalize -> END
            -> (pass_through) -> finalize -> END
    """
    builder = StateGraph(SupervisionSubgraphState)

    # Add nodes
    builder.add_node("analyze_review", analyze_review_node)
    builder.add_node("expand_topic", expand_topic_node)
    builder.add_node("integrate_content", integrate_content_node)
    builder.add_node("finalize", finalize_supervision_node)

    # Entry point
    builder.add_edge(START, "analyze_review")

    # Route based on supervisor decision
    builder.add_conditional_edges(
        "analyze_review",
        route_after_analysis,
        {
            "expand": "expand_topic",
            "finalize": "finalize",
        },
    )

    # Expansion -> Integration -> Check continue
    builder.add_edge("expand_topic", "integrate_content")

    # After integration, check if we should continue or complete
    builder.add_conditional_edges(
        "integrate_content",
        should_continue_supervision,
        {
            "continue": "analyze_review",
            "complete": "finalize",
        },
    )

    # Finalize -> END
    builder.add_edge("finalize", END)

    return builder.compile()


# Export compiled graph
supervision_subgraph = create_supervision_subgraph()


async def run_supervision(
    final_review: str,
    paper_corpus: dict[str, Any],
    paper_summaries: dict[str, Any],
    clusters: list[dict],
    quality_settings: dict[str, Any],
    input_data: dict[str, Any],
    zotero_keys: dict[str, str],
) -> dict[str, Any]:
    """Run the supervision loop on a completed literature review.

    Args:
        final_review: The literature review to analyze and improve
        paper_corpus: Existing paper corpus from main workflow
        paper_summaries: Existing paper summaries
        clusters: Thematic clusters for context
        quality_settings: Quality settings (determines max_iterations)
        input_data: Original input with topic and research questions
        zotero_keys: Existing Zotero citation keys

    Returns:
        Dictionary containing:
            - final_review_v2: The improved review after supervision
            - supervision_state: Final supervision state
            - expansions: List of expansion records
            - iterations: Number of iterations performed
            - added_papers: New papers added to corpus
            - added_summaries: New paper summaries
    """
    # Determine max iterations from quality settings
    max_stages = quality_settings.get("max_stages", 3)
    max_iterations = max_stages  # Use same as diffusion stages

    # Build initial supervision state
    initial_state = {
        # Review content
        "current_review": final_review,
        "final_review_v2": None,

        # Context from main workflow
        "input": input_data,
        "paper_corpus": paper_corpus,
        "paper_summaries": paper_summaries,
        "clusters": clusters,
        "quality_settings": quality_settings,
        "zotero_keys": zotero_keys,

        # Supervision tracking
        "iteration": 0,
        "max_iterations": max_iterations,
        "supervision_depth": 0,
        "issues_explored": [],
        "is_complete": False,

        # Outputs
        "supervision_expansions": [],
        "decision": None,
        "expansion_result": None,
    }

    logger.info(
        f"Starting supervision loop: max_iterations={max_iterations}, "
        f"review length={len(final_review)} chars"
    )

    # Run the subgraph
    final_state = await supervision_subgraph.ainvoke(initial_state)

    # Extract results
    iterations = final_state.get("iteration", 0)
    expansions = final_state.get("supervision_expansions", [])

    # Collect added papers across all expansions
    added_papers = {}
    added_summaries = {}
    for exp in expansions:
        # These are accumulated via the expansion nodes
        pass

    # Get final papers/summaries from state (they were merged during expansion)
    final_corpus = final_state.get("paper_corpus", {})
    final_summaries = final_state.get("paper_summaries", {})

    # Calculate what was added (not in original)
    added_papers = {
        doi: paper for doi, paper in final_corpus.items()
        if doi not in paper_corpus
    }
    added_summaries = {
        doi: summary for doi, summary in final_summaries.items()
        if doi not in paper_summaries
    }

    logger.info(
        f"Supervision complete: {iterations} iterations, "
        f"{len(expansions)} expansions, {len(added_papers)} new papers"
    )

    return {
        "final_review_v2": final_state.get("final_review_v2", final_review),
        "supervision_state": {
            "iteration": iterations,
            "max_iterations": max_iterations,
            "supervision_depth": 0,
            "current_review": final_state.get("current_review", ""),
            "issues_explored": final_state.get("issues_explored", []),
            "is_complete": True,
        },
        "expansions": expansions,
        "iterations": iterations,
        "added_papers": added_papers,
        "added_summaries": added_summaries,
        "completion_reason": final_state.get("completion_reason", ""),
    }
