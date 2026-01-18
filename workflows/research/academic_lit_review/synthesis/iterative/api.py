"""Convenience API for iterative synthesis subgraph."""

from typing import Any, Literal, Optional

from workflows.research.academic_lit_review.state import (
    LitReviewInput,
    PaperSummary,
    QualitySettings,
    ThematicCluster,
)
from workflows.research.academic_lit_review.clustering import ClusterAnalysis
from workflows.shared.tracing import workflow_traceable, merge_trace_config

from .types import DraftVersion, IterativeSynthesisState, MAX_ITERATIONS
from .graph import iterative_synthesis_subgraph


@workflow_traceable(name="IterativeSynthesis", workflow_type="iterative_synthesis")
async def run_iterative_synthesis(
    paper_summaries: dict[str, PaperSummary],
    clusters: list[ThematicCluster],
    cluster_analyses: list[ClusterAnalysis],
    topic: str,
    research_questions: list[str],
    quality_settings: QualitySettings,
    zotero_keys: Optional[dict[str, str]] = None,
    ordering_strategy: Literal["by_cluster", "random", "by_date", "interleaved"] = "by_cluster",
    max_iterations: int = MAX_ITERATIONS,
) -> dict[str, Any]:
    """Run iterative synthesis as a standalone operation.

    The iterative approach builds the review incrementally by incorporating
    2-3 papers at a time, emphasizing cross-theme connections throughout.

    Args:
        paper_summaries: DOI -> PaperSummary mapping
        clusters: List of ThematicClusters from clustering phase
        cluster_analyses: List of ClusterAnalysis from clustering phase
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality tier settings
        zotero_keys: Optional DOI -> Zotero key mapping (required)
        ordering_strategy: How to order papers for incorporation:
            - "by_cluster": Process papers cluster-by-cluster (default)
            - "interleaved": Round-robin across clusters for early cross-theme
            - "random": Random order (tests for ordering bias)
            - "by_date": Chronological order (historical narrative)
        max_iterations: Maximum iterations before forcing completion

    Returns:
        Dict with final_review, references, citation_keys, completeness metrics,
        draft versions, and iteration count
    """
    # Validate Zotero keys - same as map-reduce synthesis
    if zotero_keys is None:
        raise ValueError(
            "zotero_keys cannot be None - all papers must have Zotero keys from document_processing"
        )

    missing_keys = [
        doi
        for doi in paper_summaries.keys()
        if doi not in zotero_keys or not zotero_keys[doi]
    ]
    if missing_keys:
        raise ValueError(
            f"Papers missing Zotero keys: {missing_keys[:5]}"
            f"{'...' if len(missing_keys) > 5 else ''}. "
            f"Document processing may have failed for these papers."
        )

    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality="standard",
        date_range=None,
    )

    # Initialize with all papers to process
    initial_state = IterativeSynthesisState(
        input=input_data,
        quality_settings=quality_settings,
        paper_summaries=paper_summaries,
        clusters=clusters,
        cluster_analyses=cluster_analyses,
        zotero_keys=zotero_keys,
        # Iterative state
        current_draft=DraftVersion(
            content="",
            version=0,
            papers_incorporated=[],
            themes_covered=[],
            cross_references=[],
            gaps_remaining=[],
        ),
        papers_to_process=list(paper_summaries.keys()),
        papers_in_progress=[],
        ordering_strategy=ordering_strategy,
        iteration=0,
        max_iterations=max_iterations,
        completeness_score=0.0,
        # Sub-scores
        paper_incorporation_rate=0.0,
        theme_coverage_score=0.0,
        cross_reference_density=0.0,
        # Output
        final_review="",
        references=[],
        citation_keys=[],
        prisma_documentation="",
        errors=[],
    )

    # Set recursion limit high enough for all papers (4 nodes per iteration + setup)
    # Each iteration uses: select_papers -> incorporate_papers -> check_completeness -> (continue)
    # Plus: START -> initialize_draft -> ... -> finalize_draft -> END
    max_graph_steps = (max_iterations * 4) + 10
    config = merge_trace_config({"recursion_limit": max_graph_steps})

    result = await iterative_synthesis_subgraph.ainvoke(initial_state, config=config)

    return {
        "final_review": result.get("final_review", ""),
        "references": result.get("references", []),
        "citation_keys": result.get("citation_keys", []),
        # Iterative-specific outputs
        "ordering_strategy": ordering_strategy,
        "iterations": result.get("iteration", 0),
        "completeness_score": result.get("completeness_score", 0.0),
        "paper_incorporation_rate": result.get("paper_incorporation_rate", 0.0),
        "theme_coverage_score": result.get("theme_coverage_score", 0.0),
        "cross_reference_density": result.get("cross_reference_density", 0.0),
        "papers_incorporated": (
            result.get("current_draft", {}).get("papers_incorporated", [])
        ),
        "cross_references": (
            result.get("current_draft", {}).get("cross_references", [])
        ),
        "draft_versions": result.get("current_draft", {}).get("version", 0),
        "errors": result.get("errors", []),
    }
