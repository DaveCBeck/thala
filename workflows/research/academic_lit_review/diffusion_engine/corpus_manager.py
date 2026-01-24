"""Corpus and graph synchronization."""

import logging
from datetime import datetime
from typing import Any

from .types import DiffusionEngineState

logger = logging.getLogger(__name__)


async def update_corpus_and_graph(state: DiffusionEngineState) -> dict[str, Any]:
    """Add newly relevant papers to corpus and update citation graph."""
    llm_relevant = state.get("current_stage_relevant", [])
    llm_rejected = state.get("current_stage_rejected", [])
    candidates = state.get("current_stage_candidates", [])
    citation_edges = state.get("new_citation_edges", [])
    citation_graph = state.get("citation_graph")
    paper_corpus = state.get("paper_corpus", {})
    diffusion = state["diffusion"]

    # Build lookup from candidates (enriched with corpus_cocitations and relevance_score)
    candidate_lookup = {p.get("doi"): p for p in candidates if p.get("doi")}

    # Add relevant papers to corpus
    new_corpus_papers = {}
    for doi in llm_relevant:
        paper = candidate_lookup.get(doi)
        if paper:
            new_corpus_papers[doi] = paper

    # Update citation graph with new papers and edges
    if citation_graph:
        for doi, paper in new_corpus_papers.items():
            citation_graph.add_paper(doi, paper)

        for edge in citation_edges:
            citation_graph.add_citation(
                citing_doi=edge["citing_doi"],
                cited_doi=edge["cited_doi"],
                edge_type=edge["edge_type"],
            )

    total_candidates = len(llm_relevant) + len(llm_rejected)
    coverage_delta = (
        len(llm_relevant) / total_candidates if total_candidates > 0 else 0.0
    )

    current_stage = diffusion["current_stage"]
    stages = diffusion["stages"]
    if stages and stages[-1]["stage_number"] == current_stage:
        stages[-1]["new_relevant"] = llm_relevant
        stages[-1]["new_rejected"] = llm_rejected
        stages[-1]["coverage_delta"] = coverage_delta
        stages[-1]["completed_at"] = datetime.utcnow()

    new_consecutive_low = (
        diffusion["consecutive_low_coverage"] + 1
        if coverage_delta < diffusion["saturation_threshold"]
        else 0
    )

    updated_diffusion = {
        **diffusion,
        "stages": stages,
        "consecutive_low_coverage": new_consecutive_low,
        "total_papers_discovered": diffusion["total_papers_discovered"]
        + total_candidates,
        "total_papers_relevant": diffusion["total_papers_relevant"]
        + len(llm_relevant),
        "total_papers_rejected": diffusion["total_papers_rejected"] + len(llm_rejected),
    }

    updated_corpus = {**paper_corpus, **new_corpus_papers}

    logger.info(
        f"Stage {current_stage}: Added {len(new_corpus_papers)} papers to corpus "
        f"(coverage_delta={coverage_delta:.2f}). Total corpus: {len(updated_corpus)} papers"
    )

    return {
        "paper_corpus": updated_corpus,
        "citation_graph": citation_graph,
        "diffusion": updated_diffusion,
        "current_stage_candidates": [],
        "current_stage_relevant": [],
        "current_stage_rejected": [],
        "new_citation_edges": [],
    }
