"""Public API for running diffusion engine."""

import logging
from typing import Any

from workflows.academic_lit_review.state import (
    LitReviewDiffusionState,
    LitReviewInput,
    PaperMetadata,
    QualitySettings,
)
from workflows.academic_lit_review.citation_graph import CitationGraph
from .types import DiffusionEngineState
from .graph import diffusion_engine_subgraph

logger = logging.getLogger(__name__)


async def run_diffusion(
    discovery_seeds: list[str],
    paper_corpus: dict[str, PaperMetadata],
    topic: str,
    research_questions: list[str],
    quality_settings: QualitySettings,
) -> dict[str, Any]:
    """Run diffusion engine as a standalone operation.

    Args:
        discovery_seeds: DOIs from discovery phase to seed diffusion
        paper_corpus: Initial paper corpus (from discovery)
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality tier settings

    Returns:
        Dict with final_corpus_dois, paper_corpus, citation_graph, diffusion state
    """
    # Validate and provide defaults for quality_settings
    if not quality_settings:
        logger.warning("Empty quality_settings received, using defaults")
        quality_settings = {"max_stages": 3, "saturation_threshold": 0.12}
    elif "max_stages" not in quality_settings or "saturation_threshold" not in quality_settings:
        logger.warning(
            f"Incomplete quality_settings (keys: {list(quality_settings.keys())}), "
            "missing keys will use defaults"
        )

    # Initialize citation graph with corpus papers
    citation_graph = CitationGraph()
    for doi, metadata in paper_corpus.items():
        citation_graph.add_paper(doi, metadata)

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

    initial_state = DiffusionEngineState(
        input=input_data,
        quality_settings=quality_settings,
        discovery_seeds=discovery_seeds,
        citation_graph=citation_graph,
        paper_corpus=paper_corpus,
        diffusion=LitReviewDiffusionState(
            current_stage=0,
            max_stages=quality_settings.get("max_stages", 3),
            stages=[],
            saturation_threshold=quality_settings.get("saturation_threshold", 0.12),
            is_saturated=False,
            consecutive_low_coverage=0,
            total_papers_discovered=0,
            total_papers_relevant=0,
            total_papers_rejected=0,
        ),
        current_stage_seeds=[],
        current_stage_candidates=[],
        current_stage_relevant=[],
        current_stage_rejected=[],
        new_citation_edges=[],
        cocitation_included=[],
        final_corpus_dois=[],
        saturation_reason=None,
    )

    result = await diffusion_engine_subgraph.ainvoke(initial_state)

    return {
        "final_corpus_dois": result.get("final_corpus_dois", []),
        "paper_corpus": result.get("paper_corpus", {}),
        "citation_graph": result.get("citation_graph"),
        "diffusion": result.get("diffusion"),
        "saturation_reason": result.get("saturation_reason"),
    }
