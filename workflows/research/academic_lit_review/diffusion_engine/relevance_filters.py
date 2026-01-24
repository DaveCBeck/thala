"""Relevance filtering with corpus co-citation context for LLM scoring."""

import logging
from typing import Any

from workflows.research.academic_lit_review.citation_graph import CitationGraph
from workflows.research.academic_lit_review.state import FallbackCandidate
from workflows.research.academic_lit_review.utils import batch_score_relevance
from workflows.shared.llm_utils import ModelTier
from .types import DiffusionEngineState

logger = logging.getLogger(__name__)


async def enrich_with_cocitation_counts_node(
    state: DiffusionEngineState,
) -> dict[str, Any]:
    """Compute corpus co-citation counts for each candidate.

    Adds 'corpus_cocitations' field to each candidate paper, indicating how many
    papers in the current corpus cite or are cited by this paper. This count is
    passed to the LLM as additional context for relevance scoring.
    """
    candidates = state.get("current_stage_candidates", [])
    citation_graph = state.get("citation_graph")
    citation_edges = state.get("new_citation_edges", [])
    corpus_dois = set(state.get("paper_corpus", {}).keys())

    if not candidates:
        return {"current_stage_candidates": []}

    if not citation_graph:
        # No graph available - set all counts to 0
        for candidate in candidates:
            candidate["corpus_cocitations"] = 0
        return {"current_stage_candidates": candidates}

    # Build temporary graph with corpus + candidates + new edges
    temp_graph = CitationGraph()

    # Add existing corpus nodes
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

    # Compute co-citation count for each candidate
    enriched_candidates = []
    for candidate in candidates:
        doi = candidate.get("doi")
        if not doi:
            continue

        cocitation_count = temp_graph.get_corpus_overlap_count(
            paper_doi=doi,
            corpus_dois=corpus_dois,
        )
        candidate["corpus_cocitations"] = cocitation_count
        enriched_candidates.append(candidate)

    high_cocitation = sum(1 for c in enriched_candidates if c.get("corpus_cocitations", 0) >= 3)
    logger.info(
        f"Co-citation enrichment: {len(enriched_candidates)} candidates, "
        f"{high_cocitation} with 3+ corpus connections"
    )

    return {"current_stage_candidates": enriched_candidates}


async def score_relevance_node(state: DiffusionEngineState) -> dict[str, Any]:
    """Score all candidates with LLM, using corpus co-citation counts as context.

    Each candidate should have 'corpus_cocitations' field set by
    enrich_with_cocitation_counts_node. This count is passed to the LLM
    as additional context for relevance scoring.

    Returns relevant papers + fallback candidates (0.5-0.6 score) for the fallback queue.
    """
    candidates = state.get("current_stage_candidates", [])
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])
    language_config = state.get("language_config")

    if not candidates:
        logger.info("No candidates for LLM relevance scoring")
        return {
            "current_stage_relevant": [],
            "current_stage_rejected": [],
            "current_stage_fallback": [],
        }

    # Score relevance with LLM - now returns 3-tuple
    relevant, fallback_candidates, rejected = await batch_score_relevance(
        papers=candidates,
        topic=topic,
        research_questions=research_questions,
        threshold=0.6,
        fallback_threshold=0.5,
        language_config=language_config,
        tier=ModelTier.DEEPSEEK_V3,
        max_concurrent=10,
        use_batch_api=quality_settings.get("use_batch_api", True),
    )

    # Extract DOIs
    relevant_dois = [p.get("doi") for p in relevant if p.get("doi")]
    rejected_dois = [p.get("doi") for p in rejected if p.get("doi")]

    # Build fallback candidates for this stage
    stage_fallback: list[FallbackCandidate] = [
        FallbackCandidate(
            doi=p.get("doi", ""),
            relevance_score=p.get("relevance_score", 0.5),
            source="near_threshold",
        )
        for p in fallback_candidates
        if p.get("doi")
    ]

    logger.info(
        f"LLM relevance scoring: {len(relevant_dois)} relevant, "
        f"{len(stage_fallback)} fallback, {len(rejected_dois)} rejected"
    )

    return {
        "current_stage_relevant": relevant_dois,
        "current_stage_rejected": rejected_dois,
        "current_stage_fallback": stage_fallback,
    }
