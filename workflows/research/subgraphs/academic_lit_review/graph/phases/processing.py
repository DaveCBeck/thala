"""Processing phase node for academic literature review workflow."""

import logging
from typing import Any

from workflows.research.subgraphs.academic_lit_review.state import AcademicLitReviewState
from workflows.research.subgraphs.academic_lit_review.paper_processor import (
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

    topic = input_data["topic"]

    # Use filtered papers_to_process list (set by diffusion phase), not full corpus
    papers_to_process_dois = state.get("papers_to_process", list(paper_corpus.keys()))
    papers_to_process = [
        paper_corpus[doi] for doi in papers_to_process_dois if doi in paper_corpus
    ]

    logger.info(
        f"Starting processing phase for {len(papers_to_process)} papers "
        f"(filtered from {len(paper_corpus)} discovered)"
    )

    if not papers_to_process:
        logger.warning("No papers to process")
        return {
            "paper_summaries": {},
            "current_phase": "clustering",
            "current_status": "Processing skipped (no papers)",
        }

    processing_result = await run_paper_processing(
        papers=papers_to_process,
        quality_settings=quality_settings,
        topic=topic,
    )

    paper_summaries = processing_result.get("paper_summaries", {})
    zotero_keys = processing_result.get("zotero_keys", {})
    es_ids = processing_result.get("elasticsearch_ids", {})
    processed = processing_result.get("processed_dois", [])
    failed = processing_result.get("failed_dois", [])

    logger.info(
        f"Processing complete: {len(processed)} successful, {len(failed)} failed"
    )

    return {
        "paper_summaries": paper_summaries,
        "zotero_keys": zotero_keys,
        "elasticsearch_ids": es_ids,
        "papers_processed": processed,
        "papers_failed": failed,
        "current_phase": "clustering",
        "current_status": f"Processing complete: {len(processed)} papers summarized",
    }
