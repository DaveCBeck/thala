"""LangGraph node functions for paper processing."""

import asyncio
import logging
from typing import Any

from workflows.research.subgraphs.academic_lit_review.state import PaperMetadata

from .acquisition import run_paper_pipeline
from .extraction import extract_all_summaries, extract_summary_from_metadata
from .types import MAX_PAPER_PIPELINE_CONCURRENT, PaperProcessingState

logger = logging.getLogger(__name__)


async def acquire_and_process_papers_node(state: PaperProcessingState) -> dict[str, Any]:
    """Acquire and process all papers using unified pipeline.

    This node combines acquisition and processing as one operation per paper,
    which naturally rate-limits retrieval requests since processing takes time.
    """
    papers = state.get("papers_to_process", [])

    if not papers:
        logger.warning("No papers to process")
        return {
            "acquired_papers": {},
            "acquisition_failed": [],
            "processing_results": {},
            "processing_failed": [],
        }

    logger.info(f"Starting unified paper pipeline for {len(papers)} papers")

    acquired, processing_results, acquisition_failed, processing_failed = await run_paper_pipeline(
        papers=papers,
        max_concurrent=MAX_PAPER_PIPELINE_CONCURRENT,
    )

    return {
        "acquired_papers": acquired,
        "acquisition_failed": acquisition_failed,
        "processing_results": processing_results,
        "processing_failed": processing_failed,
    }


async def extract_summaries_node(state: PaperProcessingState) -> dict[str, Any]:
    """Extract structured summaries from processed papers.

    Falls back to metadata-based extraction when document processing fails.
    """
    processing_results = state.get("processing_results", {})
    processing_failed = state.get("processing_failed", [])
    papers = state.get("papers_to_process", [])

    papers_by_doi = {p.get("doi"): p for p in papers}

    summaries = {}
    es_ids = {}
    zotero_keys = {}

    if processing_results:
        full_text_summaries, full_text_es_ids, full_text_zotero_keys = await extract_all_summaries(
            processing_results=processing_results,
            papers_by_doi=papers_by_doi,
        )
        summaries.update(full_text_summaries)
        es_ids.update(full_text_es_ids)
        zotero_keys.update(full_text_zotero_keys)

    papers_needing_fallback = []
    for paper in papers:
        doi = paper.get("doi")
        if doi and doi not in summaries:
            papers_needing_fallback.append(paper)

    if papers_needing_fallback:
        logger.warning(
            f"Document processing failed for {len(papers_needing_fallback)} papers - "
            f"falling back to metadata-only extraction. "
            f"Full-text processing is preferred for higher quality summaries."
        )

        semaphore = asyncio.Semaphore(10)

        async def extract_with_limit(paper: PaperMetadata) -> tuple[str, Any]:
            async with semaphore:
                summary = await extract_summary_from_metadata(paper)
                return (paper.get("doi"), summary)

        tasks = [extract_with_limit(p) for p in papers_needing_fallback]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Metadata extraction failed: {result}")
                continue
            doi, summary = result
            if doi:
                summaries[doi] = summary
                zotero_keys[doi] = doi.replace("/", "_").replace(".", "")[:20].upper()

        logger.info(f"Metadata fallback extracted {len(summaries) - len(processing_results)} additional summaries")

    if not summaries:
        logger.warning("No summaries extracted (no processing results and metadata fallback failed)")

    return {
        "paper_summaries": summaries,
        "elasticsearch_ids": es_ids,
        "zotero_keys": zotero_keys,
    }
