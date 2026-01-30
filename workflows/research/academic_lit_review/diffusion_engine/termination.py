"""Saturation checking and diffusion finalization."""

import logging
from datetime import datetime, timezone
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
    # Collect 3x max_papers to ensure enough recent papers for recency quota
    collection_target = max_papers * 3

    # Check stopping conditions
    saturation_reason = None

    # 1. Max stages reached
    if diffusion["current_stage"] >= diffusion["max_stages"]:
        saturation_reason = f"Reached maximum stages ({diffusion['max_stages']})"

    # 2. Collection target reached (3x max_papers for recency pool)
    elif len(paper_corpus) >= collection_target:
        saturation_reason = f"Reached collection target ({collection_target} papers for {max_papers} final)"

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
        logger.debug(
            f"Continuing diffusion: stage {diffusion['current_stage']}/{diffusion['max_stages']}, "
            f"corpus size {len(paper_corpus)}/{collection_target} (target {max_papers} final)"
        )
        return {
            "diffusion": diffusion,
        }


async def finalize_diffusion(state: DiffusionEngineState) -> dict[str, Any]:
    """Finalize diffusion and filter to top papers with recency quota.

    Ensures ~25% of papers come from the past 3 years (if available) to
    balance seminal works with recent research.
    """
    paper_corpus = state.get("paper_corpus", {})
    saturation_reason = state.get("saturation_reason", "Unknown")
    quality_settings = state["quality_settings"]
    max_papers = _get_effective_max_papers(state)

    recency_years = quality_settings.get("recency_years", 3)
    recency_quota = quality_settings.get("recency_quota", 0.25)

    # If corpus is small enough, no filtering needed
    if len(paper_corpus) <= max_papers:
        final_dois = list(paper_corpus.keys())
        logger.info(
            f"Diffusion complete: {len(final_dois)} papers in final corpus. "
            f"Reason: {saturation_reason}"
        )
        return {"final_corpus_dois": final_dois}

    # Partition by recency
    current_year = datetime.now(timezone.utc).year
    cutoff_year = current_year - recency_years

    recent = [(doi, p) for doi, p in paper_corpus.items() if p.get("year", 0) >= cutoff_year]
    older = [(doi, p) for doi, p in paper_corpus.items() if p.get("year", 0) < cutoff_year]

    # Sort each by relevance
    recent.sort(key=lambda x: x[1].get("relevance_score", 0.5), reverse=True)
    older.sort(key=lambda x: x[1].get("relevance_score", 0.5), reverse=True)

    # Calculate target for recent papers
    target_recent = int(max_papers * recency_quota)

    # Select papers: recent first (up to quota), then older to fill
    recent_selected = recent[: min(target_recent, len(recent))]
    remaining_slots = max_papers - len(recent_selected)
    older_selected = older[:remaining_slots]

    # If we have extra slots and more recent papers, add them
    total_selected = len(recent_selected) + len(older_selected)
    if total_selected < max_papers and len(recent) > len(recent_selected):
        extra_needed = max_papers - total_selected
        extra = recent[len(recent_selected) : len(recent_selected) + extra_needed]
        recent_selected.extend(extra)

    final_dois = [doi for doi, _ in recent_selected] + [doi for doi, _ in older_selected]

    # Log composition
    actual_recent = len(
        [d for d in final_dois if paper_corpus[d].get("year", 0) >= cutoff_year]
    )
    recent_pct = actual_recent / len(final_dois) if final_dois else 0
    logger.info(
        f"Diffusion complete: {len(final_dois)} papers "
        f"({actual_recent} recent = {recent_pct:.0%}). Reason: {saturation_reason}"
    )

    return {"final_corpus_dois": final_dois}


def should_continue_diffusion(state: DiffusionEngineState) -> str:
    """Determine if diffusion should continue or finalize."""
    diffusion = state.get("diffusion", {})

    if diffusion.get("is_saturated", False):
        return "finalize"
    return "continue"
