"""Stage node functions: initialization, seed selection, and citation expansion."""

import logging
from datetime import datetime, timezone
from typing import Any

from workflows.research.academic_lit_review.state import (
    DiffusionStage,
    LitReviewDiffusionState,
)
from workflows.research.academic_lit_review.utils import (
    convert_to_paper_metadata,
    deduplicate_papers,
)
from .types import DiffusionEngineState
from .citation_fetcher import fetch_citations_raw

logger = logging.getLogger(__name__)


async def initialize_diffusion(state: DiffusionEngineState) -> dict[str, Any]:
    """Initialize diffusion state from discovery seeds."""
    discovery_seeds = state.get("discovery_seeds", [])
    quality_settings = state["quality_settings"]
    max_stages = quality_settings.get("max_stages", 3)
    saturation_threshold = quality_settings.get("saturation_threshold", 0.12)

    if not discovery_seeds:
        logger.warning("No discovery seeds provided for diffusion")
        return {
            "diffusion": LitReviewDiffusionState(
                current_stage=0,
                max_stages=max_stages,
                stages=[],
                saturation_threshold=saturation_threshold,
                is_saturated=True,
                consecutive_low_coverage=0,
                total_papers_discovered=0,
                total_papers_relevant=0,
                total_papers_rejected=0,
            ),
            "saturation_reason": "No discovery seeds provided",
        }

    # Initialize diffusion tracking
    diffusion_state = LitReviewDiffusionState(
        current_stage=0,
        max_stages=max_stages,
        stages=[],
        saturation_threshold=saturation_threshold,
        is_saturated=False,
        consecutive_low_coverage=0,
        total_papers_discovered=len(discovery_seeds),
        total_papers_relevant=len(discovery_seeds),
        total_papers_rejected=0,
    )

    logger.info(
        f"Initialized diffusion with {len(discovery_seeds)} seeds, "
        f"max_stages={max_stages}, "
        f"saturation_threshold={saturation_threshold}"
    )

    return {
        "diffusion": diffusion_state,
        "current_stage_seeds": [],
        "current_stage_candidates": [],
        "current_stage_relevant": [],
        "current_stage_rejected": [],
        "new_citation_edges": [],
    }


async def select_expansion_seeds(state: DiffusionEngineState) -> dict[str, Any]:
    """Select papers to expand from using citation graph analysis."""
    citation_graph = state.get("citation_graph")
    diffusion = state["diffusion"]
    quality_settings = state["quality_settings"]

    if not citation_graph or citation_graph.node_count == 0:
        logger.warning("No papers in citation graph to select seeds from")
        return {
            "current_stage_seeds": [],
            "diffusion": {**diffusion, "is_saturated": True},
            "saturation_reason": "No papers in citation graph",
        }

    # Increment stage
    new_stage = diffusion["current_stage"] + 1

    # Use citation graph to get prioritized expansion candidates
    max_seeds = min(20, quality_settings.get("max_papers", 100) // 10)
    seed_dois = citation_graph.get_expansion_candidates(
        max_papers=max_seeds,
        prioritize_recent=True,
    )

    if not seed_dois:
        logger.debug("No expansion candidates found, saturation reached")
        return {
            "current_stage_seeds": [],
            "diffusion": {**diffusion, "is_saturated": True},
            "saturation_reason": "No expansion candidates available",
        }

    # Create new stage record
    new_stage_record = DiffusionStage(
        stage_number=new_stage,
        seed_papers=seed_dois,
        forward_papers_found=0,
        backward_papers_found=0,
        new_relevant=[],
        new_rejected=[],
        coverage_delta=0.0,
        started_at=datetime.now(timezone.utc),
        completed_at=None,
    )

    updated_diffusion = {
        **diffusion,
        "current_stage": new_stage,
        "stages": diffusion["stages"] + [new_stage_record],
    }

    logger.debug(f"Stage {new_stage}: Selected {len(seed_dois)} expansion seeds")

    return {
        "current_stage_seeds": seed_dois,
        "diffusion": updated_diffusion,
    }


async def run_citation_expansion_node(state: DiffusionEngineState) -> dict[str, Any]:
    """Fetch forward and backward citations for selected seeds."""
    seed_dois = state.get("current_stage_seeds", [])
    quality_settings = state["quality_settings"]
    existing_dois = set(state.get("paper_corpus", {}).keys())

    if not seed_dois:
        logger.warning("No seeds to expand from")
        return {
            "current_stage_candidates": [],
            "new_citation_edges": [],
        }

    min_citations = quality_settings.get("min_citations_filter", 10)
    recency_years = quality_settings.get("recency_years", 3)

    # Fetch all citations with recency-aware thresholds
    raw_results, citation_edges = await fetch_citations_raw(
        seed_dois=seed_dois,
        min_citations=min_citations,
        recency_years=recency_years,
    )

    # Convert to PaperMetadata
    diffusion = state["diffusion"]
    current_stage = diffusion["current_stage"]

    candidates = []
    for result in raw_results:
        paper = convert_to_paper_metadata(
            work=result,
            discovery_stage=current_stage,
            discovery_method="diffusion",
        )
        if paper:
            candidates.append(paper)

    # Deduplicate and remove existing corpus papers
    candidates = deduplicate_papers(candidates, existing_dois)

    logger.info(
        f"Stage {current_stage}: Fetched {len(raw_results)} citations, "
        f"{len(candidates)} unique candidates after deduplication"
    )

    return {
        "current_stage_candidates": candidates,
        "new_citation_edges": citation_edges,
    }
