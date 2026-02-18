"""Processing phase node for academic literature review workflow."""

import logging
from typing import Any

from workflows.research.academic_lit_review.state import AcademicLitReviewState
from workflows.research.academic_lit_review.paper_processor import (
    run_paper_processing,
)

logger = logging.getLogger(__name__)


async def processing_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 3: Process papers for full-text extraction and summarization.

    Uses document_processing workflow for PDF handling and summary extraction.
    """
    input_data = state["input"]
    paper_corpus = state.get("paper_corpus", {})
    quality_settings = state["quality_settings"]
    fallback_queue = state.get("fallback_queue", [])

    topic = input_data["topic"]

    # Use filtered papers_to_process list (set by diffusion phase), not full corpus
    papers_to_process_dois = state.get("papers_to_process", list(paper_corpus.keys()))
    papers_to_process = [paper_corpus[doi] for doi in papers_to_process_dois if doi in paper_corpus]

    logger.info(
        f"Starting processing phase for {len(papers_to_process)} papers (filtered from {len(paper_corpus)} discovered)"
    )

    if not papers_to_process:
        logger.warning("No papers to process")
        return {
            "paper_summaries": {},
            "current_phase": "clustering",
            "current_status": "Processing skipped (no papers)",
        }

    # Get language config for verification (non-English papers)
    language_config = state.get("language_config")

    processing_result = await run_paper_processing(
        papers=papers_to_process,
        quality_settings=quality_settings,
        topic=topic,
        language_config=language_config,
        fallback_queue=fallback_queue,
        paper_corpus=paper_corpus,
    )

    paper_summaries = processing_result.get("paper_summaries", {})
    zotero_keys = processing_result.get("zotero_keys", {})
    es_ids = processing_result.get("elasticsearch_ids", {})
    processed = processing_result.get("processed_dois", [])
    failed = processing_result.get("failed_dois", [])
    fallback_substitutions = processing_result.get("fallback_substitutions", [])
    fallback_exhausted = processing_result.get("fallback_exhausted", [])

    logger.info(f"Processing complete: {len(processed)} successful, {len(failed)} failed")

    return {
        "paper_summaries": paper_summaries,
        "zotero_keys": zotero_keys,
        "elasticsearch_ids": es_ids,
        "papers_processed": processed,
        "papers_failed": failed,
        "fallback_substitutions": fallback_substitutions,
        "fallback_exhausted": fallback_exhausted,
        "current_phase": "clustering",
        "current_status": f"Processing complete: {len(processed)} papers summarized",
    }
