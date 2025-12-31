"""Clustering phase node for academic literature review workflow."""

import logging
from typing import Any

from workflows.research.subgraphs.academic_lit_review.state import AcademicLitReviewState
from workflows.research.subgraphs.academic_lit_review.clustering import (
    run_clustering,
)

logger = logging.getLogger(__name__)


async def clustering_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 4: Cluster papers into thematic groups.

    Uses dual-strategy clustering (BERTopic + LLM) with Opus synthesis.
    """
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    paper_summaries = state.get("paper_summaries", {})

    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])

    logger.info(f"Starting clustering phase for {len(paper_summaries)} papers")

    if not paper_summaries:
        logger.warning("No paper summaries to cluster")
        return {
            "clusters": [],
            "current_phase": "synthesis",
            "current_status": "Clustering skipped (no summaries)",
        }

    clustering_result = await run_clustering(
        paper_summaries=paper_summaries,
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
    )

    clusters = clustering_result.get("final_clusters", [])
    cluster_analyses = clustering_result.get("cluster_analyses", [])
    bertopic_clusters = clustering_result.get("bertopic_clusters", [])
    llm_schema = clustering_result.get("llm_topic_schema")

    logger.info(f"Clustering complete: {len(clusters)} thematic clusters identified")

    return {
        "clusters": clusters,
        "bertopic_clusters": bertopic_clusters,
        "llm_topic_schema": llm_schema,
        # Store cluster analyses in state for synthesis
        "_cluster_analyses": cluster_analyses,
        "current_phase": "synthesis",
        "current_status": f"Clustering complete: {len(clusters)} themes identified",
    }
