"""Diffusion phase node for academic literature review workflow."""

import logging
from typing import Any

from workflows.research.subgraphs.academic_lit_review.state import (
    AcademicLitReviewState,
    LitReviewDiffusionState,
)
from workflows.research.subgraphs.academic_lit_review.diffusion_engine import (
    run_diffusion,
)

logger = logging.getLogger(__name__)


async def diffusion_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 2: Expand corpus through recursive citation diffusion.

    Iteratively explores citation network until saturation.
    """
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    paper_corpus = state.get("paper_corpus", {})

    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])

    logger.info(f"Starting diffusion phase with {len(paper_corpus)} seed papers")

    # Use all current corpus papers as seeds for diffusion
    discovery_seeds = list(paper_corpus.keys())

    if not discovery_seeds:
        logger.warning("No seeds for diffusion, skipping phase")
        return {
            "diffusion": LitReviewDiffusionState(
                current_stage=0,
                max_stages=quality_settings["max_stages"],
                stages=[],
                saturation_threshold=quality_settings["saturation_threshold"],
                is_saturated=True,
                consecutive_low_coverage=0,
                total_papers_discovered=0,
                total_papers_relevant=0,
                total_papers_rejected=0,
            ),
            "current_phase": "processing",
            "current_status": "Diffusion skipped (no seeds)",
        }

    diffusion_result = await run_diffusion(
        discovery_seeds=discovery_seeds,
        paper_corpus=paper_corpus,
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
    )

    final_corpus = diffusion_result.get("paper_corpus", paper_corpus)
    final_corpus_dois = diffusion_result.get("final_corpus_dois", list(final_corpus.keys()))
    diffusion_state = diffusion_result.get("diffusion", {})
    saturation_reason = diffusion_result.get("saturation_reason", "Unknown")

    logger.info(
        f"Diffusion complete: {len(final_corpus_dois)} papers selected from "
        f"{len(final_corpus)} discovered. Reason: {saturation_reason}"
    )

    return {
        "paper_corpus": final_corpus,
        "diffusion": diffusion_state,
        "papers_to_process": final_corpus_dois,
        "current_phase": "processing",
        "current_status": f"Diffusion complete: {len(final_corpus_dois)} papers ({saturation_reason})",
    }
