"""Synthesis phase node for academic literature review workflow."""

import logging
from datetime import datetime
from typing import Any

from workflows.research.subgraphs.academic_lit_review.state import AcademicLitReviewState
from workflows.research.subgraphs.academic_lit_review.synthesis import (
    run_synthesis,
)

logger = logging.getLogger(__name__)


async def synthesis_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 5: Synthesize findings into coherent literature review.

    Writes thematic sections, integrates document, processes citations.
    """
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    paper_summaries = state.get("paper_summaries", {})
    clusters = state.get("clusters", [])
    zotero_keys = state.get("zotero_keys", {})

    # Retrieve cluster analyses from state
    cluster_analyses = state.get("_cluster_analyses", [])

    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])

    logger.info(f"Starting synthesis phase with {len(clusters)} clusters")

    if not clusters or not paper_summaries:
        logger.warning("No clusters or summaries for synthesis")
        return {
            "final_review": "Literature review generation failed: no content available.",
            "current_phase": "complete",
            "current_status": "Synthesis skipped (no content)",
            "completed_at": datetime.utcnow(),
        }

    synthesis_result = await run_synthesis(
        paper_summaries=paper_summaries,
        clusters=clusters,
        cluster_analyses=cluster_analyses,
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
        zotero_keys=zotero_keys,
    )

    final_review = synthesis_result.get("final_review", "")
    references = synthesis_result.get("references", [])
    quality_metrics = synthesis_result.get("quality_metrics", {})
    quality_passed = synthesis_result.get("quality_passed", False)
    prisma_docs = synthesis_result.get("prisma_documentation", "")

    logger.info(
        f"Synthesis complete: {quality_metrics.get('total_words', 0)} words, "
        f"{quality_metrics.get('unique_papers_cited', 0)} papers cited, "
        f"quality_passed={quality_passed}"
    )

    return {
        "final_review": final_review,
        "references": references,
        "prisma_documentation": prisma_docs,
        "section_drafts": synthesis_result.get("section_drafts", {}),
        "current_phase": "complete",
        "current_status": f"Complete: {quality_metrics.get('total_words', 0)} word review",
        "completed_at": datetime.utcnow(),
    }
