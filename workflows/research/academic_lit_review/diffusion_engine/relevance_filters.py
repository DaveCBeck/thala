"""Two-stage relevance filtering: co-citation analysis + LLM scoring."""

import logging
from typing import Any

from workflows.research.academic_lit_review.citation_graph import CitationGraph
from workflows.research.academic_lit_review.utils import batch_score_relevance
from workflows.shared.llm_utils import ModelTier
from .types import DiffusionEngineState, COCITATION_THRESHOLD

logger = logging.getLogger(__name__)


async def check_cocitation_relevance_node(
    state: DiffusionEngineState,
) -> dict[str, Any]:
    """Two-stage relevance: Stage 1 - auto-include papers co-cited with 3+ corpus papers."""
    candidates = state.get("current_stage_candidates", [])
    citation_graph = state.get("citation_graph")
    citation_edges = state.get("new_citation_edges", [])
    corpus_dois = set(state.get("paper_corpus", {}).keys())

    if not candidates or not citation_graph:
        return {
            "cocitation_included": [],
            "current_stage_candidates": candidates,
        }

    # Temporarily add new edges to graph for co-citation analysis
    temp_graph = CitationGraph()

    # Add existing nodes
    for doi, metadata in state.get("paper_corpus", {}).items():
        temp_graph.add_paper(doi, metadata)

    # Add candidate nodes
    for candidate in candidates:
        doi = candidate.get("doi")
        if doi:
            temp_graph.add_paper(doi, candidate)

    # Add all edges (existing + new)
    for edge in citation_edges:
        temp_graph.add_citation(
            citing_doi=edge["citing_doi"],
            cited_doi=edge["cited_doi"],
            edge_type=edge["edge_type"],
        )

    # Check co-citation for each candidate
    cocitation_included = []
    remaining_candidates = []

    for candidate in candidates:
        doi = candidate.get("doi")
        if not doi:
            continue

        # Check if this candidate is co-cited with corpus papers
        is_cocited = temp_graph.get_cocitation_candidates(
            paper_doi=doi,
            corpus_dois=corpus_dois,
            threshold=COCITATION_THRESHOLD,
        )

        if is_cocited:
            cocitation_included.append(doi)
            logger.debug(
                f"Auto-included via co-citation: {candidate.get('title', 'Unknown')[:50]}"
            )
        else:
            remaining_candidates.append(candidate)

    logger.info(
        f"Co-citation filter: {len(cocitation_included)} auto-included, "
        f"{len(remaining_candidates)} require LLM scoring"
    )

    return {
        "cocitation_included": cocitation_included,
        "current_stage_candidates": remaining_candidates,
    }


async def score_remaining_relevance_node(state: DiffusionEngineState) -> dict[str, Any]:
    """Two-stage relevance: Stage 2 - LLM scoring for remaining candidates."""
    candidates = state.get("current_stage_candidates", [])
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])
    language_config = state.get("language_config")

    if not candidates:
        logger.info("No candidates remaining for LLM relevance scoring")
        return {
            "current_stage_relevant": [],
            "current_stage_rejected": [],
        }

    # Score relevance with LLM
    relevant, rejected = await batch_score_relevance(
        papers=candidates,
        topic=topic,
        research_questions=research_questions,
        threshold=0.6,
        language_config=language_config,
        tier=ModelTier.DEEPSEEK_V3,
        max_concurrent=10,
        use_batch_api=quality_settings.get("use_batch_api", True),
    )

    # Extract DOIs
    relevant_dois = [p.get("doi") for p in relevant if p.get("doi")]
    rejected_dois = [p.get("doi") for p in rejected if p.get("doi")]

    logger.info(
        f"LLM relevance scoring: {len(relevant_dois)} relevant, {len(rejected_dois)} rejected"
    )

    return {
        "current_stage_relevant": relevant_dois,
        "current_stage_rejected": rejected_dois,
    }
