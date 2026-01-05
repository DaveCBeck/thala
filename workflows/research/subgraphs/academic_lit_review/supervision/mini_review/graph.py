"""Mini-review graph for Loop 2 literature base expansion.

Runs a complete literature review workflow WITHOUT supervision on a specific
literature base, excluding DOIs already in parent corpus.
"""

import logging
from typing import Any

from workflows.research.subgraphs.academic_lit_review.keyword_search import (
    run_keyword_search,
)
from workflows.research.subgraphs.academic_lit_review.diffusion_engine.api import (
    run_diffusion,
)
from workflows.research.subgraphs.academic_lit_review.paper_processor.api import (
    run_paper_processing,
)
from workflows.research.subgraphs.academic_lit_review.clustering.api import (
    run_clustering,
)
from workflows.research.subgraphs.academic_lit_review.synthesis.api import (
    run_synthesis,
)
from workflows.research.subgraphs.academic_lit_review.state import (
    PaperMetadata,
    QualitySettings,
)
from ..types import LiteratureBase

logger = logging.getLogger(__name__)


async def run_mini_review(
    literature_base: LiteratureBase,
    parent_topic: str,
    quality_settings: QualitySettings,
    exclude_dois: set[str],
) -> dict[str, Any]:
    """Run complete mini-review on a literature base without supervision.

    Executes full review pipeline: keyword search -> diffusion -> processing ->
    clustering -> synthesis, excluding DOIs already in parent corpus.

    Args:
        literature_base: Literature base specification from Loop 2 analyzer
        parent_topic: Topic of parent review (for context)
        quality_settings: Quality tier settings
        exclude_dois: DOIs already in parent corpus (to exclude)

    Returns:
        Dict containing:
            - mini_review_text: Complete review text
            - paper_summaries: DOI -> PaperSummary mapping
            - zotero_keys: DOI -> Zotero key mapping
            - clusters: List of ThematicClusters
            - references: List of FormattedCitations
    """
    logger.info(
        f"Starting mini-review for literature base: {literature_base.name} "
        f"(excluding {len(exclude_dois)} parent DOIs)"
    )

    # Keyword search
    search_topic = f"{parent_topic} - {literature_base.name} perspective"
    keyword_result = await run_keyword_search(
        topic=search_topic,
        research_questions=literature_base.search_queries,
        quality_settings=quality_settings,
        date_range=None,
        focus_areas=None,
        language_config=None,
    )

    discovered_papers = keyword_result.get("discovered_papers", [])

    # Filter out parent corpus DOIs
    filtered_papers = [
        p for p in discovered_papers
        if p.get("doi") and p["doi"] not in exclude_dois
    ]
    logger.info(
        f"Keyword search: {len(discovered_papers)} found, "
        f"{len(filtered_papers)} after filtering"
    )

    if not filtered_papers:
        logger.warning("No new papers found after filtering, returning empty review")
        return {
            "mini_review_text": "",
            "paper_summaries": {},
            "zotero_keys": {},
            "clusters": [],
            "references": [],
        }

    # Convert to PaperMetadata dict
    initial_corpus: dict[str, PaperMetadata] = {
        p["doi"]: p for p in filtered_papers
    }
    seed_dois = list(initial_corpus.keys())

    # Diffusion
    diffusion_result = await run_diffusion(
        discovery_seeds=seed_dois,
        paper_corpus=initial_corpus,
        topic=search_topic,
        research_questions=literature_base.search_queries,
        quality_settings=quality_settings,
    )

    final_corpus = diffusion_result.get("paper_corpus", initial_corpus)

    # Filter diffusion results to exclude parent DOIs
    final_corpus = {
        doi: metadata
        for doi, metadata in final_corpus.items()
        if doi not in exclude_dois
    }
    logger.info(f"Diffusion: {len(final_corpus)} papers after filtering")

    if not final_corpus:
        logger.warning("No papers after diffusion filtering")
        return {
            "mini_review_text": "",
            "paper_summaries": {},
            "zotero_keys": {},
            "clusters": [],
            "references": [],
        }

    # Paper processing
    papers_to_process = list(final_corpus.values())
    processing_result = await run_paper_processing(
        papers=papers_to_process,
        quality_settings=quality_settings,
        topic=search_topic,
    )

    paper_summaries = processing_result.get("paper_summaries", {})
    zotero_keys = processing_result.get("zotero_keys", {})
    logger.info(f"Processing: {len(paper_summaries)} papers processed")

    if not paper_summaries:
        logger.warning("No papers successfully processed")
        return {
            "mini_review_text": "",
            "paper_summaries": {},
            "zotero_keys": {},
            "clusters": [],
            "references": [],
        }

    # Clustering
    clustering_result = await run_clustering(
        paper_summaries=paper_summaries,
        topic=search_topic,
        research_questions=literature_base.search_queries,
        quality_settings=quality_settings,
    )

    clusters = clustering_result.get("final_clusters", [])
    cluster_analyses = clustering_result.get("cluster_analyses", [])
    logger.info(f"Clustering: {len(clusters)} clusters identified")

    # Synthesis
    synthesis_result = await run_synthesis(
        paper_summaries=paper_summaries,
        clusters=clusters,
        cluster_analyses=cluster_analyses,
        topic=search_topic,
        research_questions=literature_base.search_queries,
        quality_settings=quality_settings,
        zotero_keys=zotero_keys,
    )

    mini_review_text = synthesis_result.get("final_review", "")
    references = synthesis_result.get("references", [])

    logger.info(
        f"Mini-review complete: {len(mini_review_text)} chars, "
        f"{len(paper_summaries)} papers, {len(references)} references"
    )

    return {
        "mini_review_text": mini_review_text,
        "paper_summaries": paper_summaries,
        "zotero_keys": zotero_keys,
        "clusters": clusters,
        "references": references,
    }
