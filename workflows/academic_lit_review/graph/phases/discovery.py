"""Discovery phase node for academic literature review workflow."""

import logging
from typing import Any

from workflows.academic_lit_review.state import AcademicLitReviewState
from workflows.academic_lit_review.keyword_search import (
    run_keyword_search,
)
from workflows.academic_lit_review.citation_network import (
    run_citation_expansion,
)

logger = logging.getLogger(__name__)


async def discovery_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 1: Discover seed papers through multiple strategies.

    Runs keyword search and initial citation network discovery in parallel.
    """
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    language_config = state.get("language_config")

    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])
    date_range = input_data.get("date_range")
    focus_areas = input_data.get("focus_areas")

    logger.info(f"Starting discovery phase for: {topic}")

    # Run keyword search
    keyword_result = await run_keyword_search(
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
        date_range=date_range,
        focus_areas=focus_areas,
        language_config=language_config,
    )

    keyword_papers = keyword_result.get("discovered_papers", [])
    keyword_dois = keyword_result.get("keyword_dois", [])

    # Build initial paper corpus from keyword search
    paper_corpus = {}
    for paper in keyword_papers:
        doi = paper.get("doi")
        if doi:
            paper_corpus[doi] = paper

    # Run initial citation expansion on top keyword results
    top_seed_dois = keyword_dois[:10]  # Use top 10 as seeds

    if top_seed_dois:
        citation_result = await run_citation_expansion(
            seed_dois=top_seed_dois,
            topic=topic,
            research_questions=research_questions,
            quality_settings=quality_settings,
            existing_dois=set(paper_corpus.keys()),
            language_config=language_config,
        )

        citation_papers = citation_result.get("discovered_papers", [])
        citation_dois = citation_result.get("citation_dois", [])

        # Add citation-discovered papers to corpus
        for paper in citation_papers:
            doi = paper.get("doi")
            if doi and doi not in paper_corpus:
                paper_corpus[doi] = paper
    else:
        citation_dois = []

    logger.info(
        f"Discovery complete: {len(keyword_dois)} from keywords, "
        f"{len(citation_dois)} from citations, {len(paper_corpus)} total"
    )

    return {
        "keyword_papers": keyword_dois,
        "citation_papers": citation_dois,
        "paper_corpus": paper_corpus,
        "current_phase": "diffusion",
        "current_status": f"Discovery complete: {len(paper_corpus)} papers found",
    }
