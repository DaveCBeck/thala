"""Public API for standalone clustering execution."""

from typing import Any

from workflows.academic_lit_review.state import (
    LitReviewInput,
    PaperSummary,
    QualitySettings,
)

from .analysis import ClusterAnalysis
from .graph import ClusteringState, clustering_subgraph


async def run_clustering(
    paper_summaries: dict[str, PaperSummary],
    topic: str,
    research_questions: list[str],
    quality_settings: QualitySettings,
) -> dict[str, Any]:
    """Run dual-strategy clustering as a standalone operation.

    Args:
        paper_summaries: DOI -> PaperSummary mapping
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality tier settings

    Returns:
        Dict with final_clusters, cluster_labels, cluster_analyses,
        and intermediate results (bertopic_clusters, llm_topic_schema)
    """
    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality="standard",
        date_range=None,
        include_books=False,
        focus_areas=None,
        exclude_terms=None,
        max_papers=None,
    )

    initial_state = ClusteringState(
        input=input_data,
        quality_settings=quality_settings,
        paper_summaries=paper_summaries,
        document_texts=[],
        document_dois=[],
        bertopic_clusters=None,
        bertopic_error=None,
        llm_topic_schema=None,
        llm_error=None,
        final_clusters=[],
        cluster_labels={},
        cluster_analyses=[],
    )

    result = await clustering_subgraph.ainvoke(initial_state)

    return {
        "final_clusters": result.get("final_clusters", []),
        "cluster_labels": result.get("cluster_labels", {}),
        "cluster_analyses": result.get("cluster_analyses", []),
        "bertopic_clusters": result.get("bertopic_clusters", []),
        "llm_topic_schema": result.get("llm_topic_schema"),
        "bertopic_error": result.get("bertopic_error"),
        "llm_error": result.get("llm_error"),
    }
