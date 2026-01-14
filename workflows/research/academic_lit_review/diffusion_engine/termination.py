"""Saturation checking and diffusion finalization."""

import logging
from typing import Any

from .types import DiffusionEngineState, NON_ENGLISH_PAPER_OVERHEAD

logger = logging.getLogger(__name__)


def _get_effective_max_papers(state: DiffusionEngineState) -> int:
    """Get effective max_papers, accounting for non-English language overhead.

    For non-English languages, we request more papers because some will be
    filtered out by language verification. The final max_papers limit is
    applied after verification in the paper processor.
    """
    quality_settings = state["quality_settings"]
    max_papers = quality_settings.get("max_papers", 100)

    language_config = state.get("language_config")
    if language_config and language_config.get("code") != "en":
        effective_max = int(max_papers * NON_ENGLISH_PAPER_OVERHEAD)
        logger.debug(
            f"Non-English mode: effective max_papers={effective_max} "
            f"(base={max_papers}, overhead={NON_ENGLISH_PAPER_OVERHEAD})"
        )
        return effective_max

    return max_papers


async def check_saturation_node(state: DiffusionEngineState) -> dict[str, Any]:
    """Check if diffusion should stop based on saturation conditions."""
    diffusion = state["diffusion"]
    paper_corpus = state.get("paper_corpus", {})
    max_papers = _get_effective_max_papers(state)

    # Check stopping conditions
    saturation_reason = None

    # 1. Max stages reached
    if diffusion["current_stage"] >= diffusion["max_stages"]:
        saturation_reason = f"Reached maximum stages ({diffusion['max_stages']})"

    # 2. Max papers reached
    elif len(paper_corpus) >= max_papers:
        saturation_reason = f"Reached maximum papers ({max_papers})"

    # 3. Consecutive low coverage (2 stages with delta < threshold)
    elif diffusion["consecutive_low_coverage"] >= 2:
        saturation_reason = (
            f"Low coverage for {diffusion['consecutive_low_coverage']} consecutive stages "
            f"(threshold={diffusion['saturation_threshold']})"
        )

    if saturation_reason:
        logger.info(f"Diffusion saturation: {saturation_reason}")
        return {
            "diffusion": {**diffusion, "is_saturated": True},
            "saturation_reason": saturation_reason,
        }
    else:
        logger.info(
            f"Continuing diffusion: stage {diffusion['current_stage']}/{diffusion['max_stages']}, "
            f"corpus size {len(paper_corpus)}/{max_papers}"
        )
        return {
            "diffusion": diffusion,
        }


async def finalize_diffusion(state: DiffusionEngineState) -> dict[str, Any]:
    """Finalize diffusion and filter to top papers by relevance."""
    paper_corpus = state.get("paper_corpus", {})
    saturation_reason = state.get("saturation_reason", "Unknown")
    max_papers = _get_effective_max_papers(state)

    # Filter to top N papers by relevance score if we exceeded max_papers
    if len(paper_corpus) > max_papers:
        sorted_papers = sorted(
            paper_corpus.items(),
            key=lambda x: x[1].get("relevance_score", 0.5),
            reverse=True,
        )
        cutoff_score = sorted_papers[max_papers - 1][1].get("relevance_score", 0.5)
        final_dois = [doi for doi, _ in sorted_papers[:max_papers]]
        logger.info(
            f"Diffusion complete: Filtered {len(paper_corpus)} papers to {max_papers} "
            f"(relevance cutoff: {cutoff_score:.2f}). Reason: {saturation_reason}"
        )
    else:
        final_dois = list(paper_corpus.keys())
        logger.info(
            f"Diffusion complete: {len(final_dois)} papers in final corpus. "
            f"Reason: {saturation_reason}"
        )

    return {
        "final_corpus_dois": final_dois,
    }


def should_continue_diffusion(state: DiffusionEngineState) -> str:
    """Determine if diffusion should continue or finalize."""
    diffusion = state.get("diffusion", {})

    if diffusion.get("is_saturated", False):
        return "finalize"
    return "continue"
