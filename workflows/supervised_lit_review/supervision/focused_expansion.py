"""Focused expansion for supervision loop.

Runs discovery, diffusion, and processing phases on a specific topic
identified by the supervisor, inheriting quality settings from the parent workflow.
"""

import logging
from typing import Any

from workflows.academic_lit_review.keyword_search import (
    run_keyword_search,
)
from workflows.academic_lit_review.diffusion_engine import (
    run_diffusion,
)
from workflows.academic_lit_review.paper_processor import (
    run_paper_processing,
)

logger = logging.getLogger(__name__)


async def run_focused_expansion(
    topic: str,
    research_query: str,
    quality_settings: dict[str, Any],
    exclude_dois: set[str],
    parent_topic: str,
) -> dict[str, Any]:
    """Run focused literature discovery on a specific theoretical topic.

    This function runs only the discovery, diffusion, and processing phases
    (NOT clustering or synthesis) to gather papers on a specific topic
    identified by the supervisor.

    Args:
        topic: The specific theory/concept to explore
        research_query: Search query for discovery
        quality_settings: Quality settings inherited from parent workflow
        exclude_dois: DOIs already in parent corpus (to avoid duplicates)
        parent_topic: The original research topic (for context)

    Returns:
        Dictionary containing:
            - paper_corpus: New papers discovered (excluding already-known ones)
            - paper_summaries: Summaries of processed papers
            - zotero_keys: Citation keys for new papers
            - processed_dois: Successfully processed DOIs
            - failed_dois: DOIs that failed processing
    """
    logger.info(f"Starting focused expansion on: {topic}")
    logger.info(f"Using query: {research_query}")
    logger.info(f"Excluding {len(exclude_dois)} existing DOIs")

    # Build research questions from the topic
    research_questions = [
        f"What are the key theoretical foundations of {topic}?",
        f"How does {topic} relate to {parent_topic}?",
    ]

    # Phase 1: Discovery - run keyword search on the specific topic
    logger.info("Phase 1: Running keyword search for focused topic")
    keyword_result = await run_keyword_search(
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
        date_range=None,  # No date restriction for theoretical foundations
        focus_areas=None,
    )

    discovered_papers = keyword_result.get("discovered_papers", [])
    keyword_dois = keyword_result.get("keyword_dois", [])

    # Build paper corpus, excluding already-known papers
    paper_corpus = {}
    for paper in discovered_papers:
        doi = paper.get("doi")
        if doi and doi not in exclude_dois:
            paper_corpus[doi] = paper

    logger.info(
        f"Discovery found {len(keyword_dois)} papers, "
        f"{len(paper_corpus)} are new (not in parent corpus)"
    )

    if not paper_corpus:
        logger.warning("No new papers discovered, returning empty result")
        return {
            "paper_corpus": {},
            "paper_summaries": {},
            "zotero_keys": {},
            "processed_dois": [],
            "failed_dois": [],
        }

    # Phase 2: Diffusion - expand citation network (with inherited quality settings)
    logger.info("Phase 2: Running diffusion on new discoveries")
    discovery_seeds = list(paper_corpus.keys())

    diffusion_result = await run_diffusion(
        discovery_seeds=discovery_seeds,
        paper_corpus=paper_corpus,
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
    )

    # Get filtered corpus from diffusion (respects max_papers limit)
    expanded_corpus = diffusion_result.get("paper_corpus", paper_corpus)
    final_corpus_dois = diffusion_result.get("final_corpus_dois", list(expanded_corpus.keys()))

    # Filter to only new papers (not in parent corpus) AND in the filtered list
    final_corpus = {
        doi: expanded_corpus[doi]
        for doi in final_corpus_dois
        if doi in expanded_corpus and doi not in exclude_dois
    }

    logger.info(
        f"Diffusion expanded to {len(expanded_corpus)} papers, "
        f"filtered to {len(final_corpus_dois)}, {len(final_corpus)} are new"
    )

    if not final_corpus:
        logger.warning("All diffusion papers already in corpus, returning empty result")
        return {
            "paper_corpus": {},
            "paper_summaries": {},
            "zotero_keys": {},
            "processed_dois": [],
            "failed_dois": [],
        }

    # Phase 3: Processing - extract and summarize papers
    logger.info(f"Phase 3: Processing {len(final_corpus)} new papers")
    papers_to_process = list(final_corpus.values())

    processing_result = await run_paper_processing(
        papers=papers_to_process,
        quality_settings=quality_settings,
        topic=topic,
    )

    paper_summaries = processing_result.get("paper_summaries", {})
    zotero_keys = processing_result.get("zotero_keys", {})
    processed_dois = processing_result.get("processed_dois", [])
    failed_dois = processing_result.get("failed_dois", [])

    logger.info(
        f"Focused expansion complete: {len(processed_dois)} papers processed, "
        f"{len(failed_dois)} failed"
    )

    return {
        "paper_corpus": final_corpus,
        "paper_summaries": paper_summaries,
        "zotero_keys": zotero_keys,
        "processed_dois": processed_dois,
        "failed_dois": failed_dois,
    }
