"""Corpus and graph synchronization with DOI fallback."""

import logging
from datetime import datetime
from typing import Any

from langchain_tools.openalex import get_works_by_dois
from workflows.research.subgraphs.academic_lit_review.utils import convert_to_paper_metadata
from .types import DiffusionEngineState

logger = logging.getLogger(__name__)


async def update_corpus_and_graph(state: DiffusionEngineState) -> dict[str, Any]:
    """Add newly relevant papers to corpus and update citation graph."""
    cocitation_included = state.get("cocitation_included", [])
    llm_relevant = state.get("current_stage_relevant", [])
    llm_rejected = state.get("current_stage_rejected", [])
    candidates = state.get("current_stage_candidates", [])
    citation_edges = state.get("new_citation_edges", [])
    citation_graph = state.get("citation_graph")
    paper_corpus = state.get("paper_corpus", {})
    diffusion = state["diffusion"]

    # Combine all relevant DOIs
    all_relevant_dois = set(cocitation_included) | set(llm_relevant)

    # Build lookup for candidates
    candidate_lookup = {p.get("doi"): p for p in candidates if p.get("doi")}

    # Also include co-citation candidates (need to rebuild from all candidates)
    all_candidates_lookup = {
        p.get("doi"): p
        for p in state.get("current_stage_candidates", [])
        if p.get("doi")
    }

    # Find co-cited papers not in candidates - these need DOI lookup fallback
    missing_cocited_dois = [
        doi for doi in cocitation_included
        if doi not in candidate_lookup and doi not in all_candidates_lookup
    ]

    # Fallback: fetch missing co-cited papers by DOI from OpenAlex
    fallback_papers = {}
    if missing_cocited_dois:
        logger.info(
            f"Fetching {len(missing_cocited_dois)} co-cited papers by DOI fallback"
        )
        try:
            fetched_works = await get_works_by_dois(missing_cocited_dois)
            for work in fetched_works:
                if work.doi:
                    doi_clean = work.doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
                    paper = convert_to_paper_metadata(
                        work.model_dump(),
                        discovery_stage=diffusion["current_stage"],
                        discovery_method="cocitation_fallback",
                    )
                    if paper:
                        fallback_papers[doi_clean] = paper
                        logger.debug(f"Recovered co-cited paper via DOI: {work.title[:50]}...")

            # Log any still-missing papers
            still_missing = set(missing_cocited_dois) - set(fallback_papers.keys())
            if still_missing:
                for doi in still_missing:
                    logger.warning(f"Co-cited paper {doi} not found in OpenAlex")
        except Exception as e:
            logger.warning(f"Failed to fetch co-cited papers by DOI: {e}")

    # Add relevant papers to corpus
    new_corpus_papers = {}
    for doi in all_relevant_dois:
        # Try to find in candidate lookup first, then all candidates, then fallback
        paper = (
            candidate_lookup.get(doi)
            or all_candidates_lookup.get(doi)
            or fallback_papers.get(doi)
        )
        if paper:
            # Ensure papers have relevance scores (co-citation papers bypass LLM scoring)
            if paper.get("relevance_score") is None:
                # High default for co-citation papers (included via citation network evidence)
                paper["relevance_score"] = 0.8
            new_corpus_papers[doi] = paper

    # Add papers to citation graph
    if citation_graph:
        for doi, paper in new_corpus_papers.items():
            citation_graph.add_paper(doi, paper)

        # Add citation edges
        for edge in citation_edges:
            citation_graph.add_citation(
                citing_doi=edge["citing_doi"],
                cited_doi=edge["cited_doi"],
                edge_type=edge["edge_type"],
            )

    # Calculate coverage delta
    total_candidates = len(cocitation_included) + len(llm_relevant) + len(llm_rejected)
    coverage_delta = len(all_relevant_dois) / total_candidates if total_candidates > 0 else 0.0

    # Update stage record
    current_stage = diffusion["current_stage"]
    stages = diffusion["stages"]
    if stages and stages[-1]["stage_number"] == current_stage:
        stages[-1]["new_relevant"] = list(all_relevant_dois)
        stages[-1]["new_rejected"] = llm_rejected
        stages[-1]["coverage_delta"] = coverage_delta
        stages[-1]["completed_at"] = datetime.utcnow()

    # Update diffusion tracking
    new_consecutive_low = (
        diffusion["consecutive_low_coverage"] + 1
        if coverage_delta < diffusion["saturation_threshold"]
        else 0
    )

    updated_diffusion = {
        **diffusion,
        "stages": stages,
        "consecutive_low_coverage": new_consecutive_low,
        "total_papers_discovered": diffusion["total_papers_discovered"] + total_candidates,
        "total_papers_relevant": diffusion["total_papers_relevant"] + len(all_relevant_dois),
        "total_papers_rejected": diffusion["total_papers_rejected"] + len(llm_rejected),
    }

    # Merge with existing corpus
    updated_corpus = {**paper_corpus, **new_corpus_papers}

    logger.info(
        f"Stage {current_stage}: Added {len(new_corpus_papers)} papers to corpus "
        f"(coverage_delta={coverage_delta:.2f}). Total corpus: {len(updated_corpus)} papers"
    )

    return {
        "paper_corpus": updated_corpus,
        "citation_graph": citation_graph,
        "diffusion": updated_diffusion,
        # Reset working state for next iteration
        "current_stage_candidates": [],
        "current_stage_relevant": [],
        "current_stage_rejected": [],
        "new_citation_edges": [],
        "cocitation_included": [],
    }
