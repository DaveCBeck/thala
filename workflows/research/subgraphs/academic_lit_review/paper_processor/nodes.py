"""LangGraph node functions for paper processing.

Uses Anthropic Batch API for 50% cost reduction when processing 5+ papers.
"""

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from workflows.research.subgraphs.academic_lit_review.state import PaperMetadata, PaperSummary


class MetadataSummarySchema(BaseModel):
    """Schema for metadata-based paper summary extraction."""

    key_findings: list[str] = Field(
        default_factory=list,
        description="2-3 inferred findings based on the abstract",
    )
    methodology: str = Field(
        default="Not available from abstract",
        description="Inferred methodology from the abstract (1-2 sentences)",
    )
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    themes: list[str] = Field(
        default_factory=list,
        description="3-5 topic tags based on title and abstract",
    )

from .acquisition import run_paper_pipeline
from .extraction import (
    extract_all_summaries,
    extract_summary_from_metadata,
    METADATA_SUMMARY_EXTRACTION_SYSTEM,
)
from .types import MAX_PAPER_PIPELINE_CONCURRENT, PaperProcessingState
from workflows.shared.batch_processor import BatchProcessor
from workflows.shared.llm_utils import ModelTier

logger = logging.getLogger(__name__)


async def acquire_and_process_papers_node(state: PaperProcessingState) -> dict[str, Any]:
    """Acquire and process all papers using unified pipeline.

    This node combines acquisition and processing as one operation per paper,
    which naturally rate-limits retrieval requests since processing takes time.
    """
    papers = state.get("papers_to_process", [])
    quality_settings = state["quality_settings"]
    use_batch_api = quality_settings.get("use_batch_api", True)

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
        use_batch_api=use_batch_api,
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
    Uses Anthropic Batch API for 50% cost reduction when processing 5+ papers.
    """
    processing_results = state.get("processing_results", {})
    processing_failed = state.get("processing_failed", [])
    papers = state.get("papers_to_process", [])
    quality_settings = state["quality_settings"]
    use_batch_api = quality_settings.get("use_batch_api", True)

    papers_by_doi = {p.get("doi"): p for p in papers}

    summaries = {}
    es_ids = {}
    zotero_keys = {}

    extraction_failed_dois = set()
    if processing_results:
        full_text_summaries, full_text_es_ids, full_text_zotero_keys, extraction_failed_dois = await extract_all_summaries(
            processing_results=processing_results,
            papers_by_doi=papers_by_doi,
            use_batch_api=use_batch_api,
        )
        summaries.update(full_text_summaries)
        es_ids.update(full_text_es_ids)
        zotero_keys.update(full_text_zotero_keys)

    # Papers needing fallback: those not in summaries OR those that explicitly failed extraction
    # This ensures papers that had processing_results but failed content fetch still get metadata fallback
    papers_needing_fallback = []
    for paper in papers:
        doi = paper.get("doi")
        if doi and (doi not in summaries or doi in extraction_failed_dois):
            papers_needing_fallback.append(paper)

    # Build a mapping of existing short_summaries from processing_results
    # These are generated from full-text and are preferable to abstract-based summaries
    existing_short_summaries = {}
    for doi, result in processing_results.items():
        short_summary = result.get("short_summary", "")
        if short_summary:
            existing_short_summaries[doi] = short_summary

    if papers_needing_fallback:
        logger.warning(
            f"Document processing failed for {len(papers_needing_fallback)} papers - "
            f"falling back to metadata-only extraction. "
            f"Full-text processing is preferred for higher quality summaries."
        )

        # Use batch API for 5+ papers (50% cost reduction) when enabled
        if use_batch_api and len(papers_needing_fallback) >= 5:
            fallback_summaries = await _extract_metadata_summaries_batched(
                papers_needing_fallback, existing_short_summaries
            )
            for doi, summary in fallback_summaries.items():
                summaries[doi] = summary
                zotero_keys[doi] = doi.replace("/", "_").replace(".", "")[:20].upper()
        else:
            # Fall back to concurrent calls for small batches
            semaphore = asyncio.Semaphore(10)

            async def extract_with_limit(paper: PaperMetadata) -> tuple[str, Any]:
                async with semaphore:
                    doi = paper.get("doi")
                    existing_summary = existing_short_summaries.get(doi) if doi else None
                    summary = await extract_summary_from_metadata(paper, existing_summary)
                    return (doi, summary)

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


async def _extract_metadata_summaries_batched(
    papers: list[PaperMetadata],
    existing_short_summaries: dict[str, str] | None = None,
) -> dict[str, PaperSummary]:
    """Extract metadata summaries using Anthropic Batch API for 50% cost reduction.

    Uses tool calling to guarantee valid JSON responses.

    Args:
        papers: List of paper metadata to process
        existing_short_summaries: Optional mapping of DOI -> short_summary from
            document processing (preferred over generating from abstract)
    """
    existing_short_summaries = existing_short_summaries or {}
    processor = BatchProcessor(poll_interval=30)

    # Define tool for structured output
    extraction_tool = {
        "name": "extract_metadata",
        "description": "Extract structured metadata from an academic paper's abstract",
        "input_schema": MetadataSummarySchema.model_json_schema(),
    }

    paper_index = {}  # Map custom_id back to paper
    for i, paper in enumerate(papers):
        doi = paper.get("doi", f"unknown-{i}")
        custom_id = f"meta-{i}"
        paper_index[custom_id] = paper

        title = paper.get("title", "Unknown Title")
        abstract = paper.get("abstract", "")
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        authors_str = ", ".join(authors[:5])
        if len(authors) > 5:
            authors_str += " et al."

        user_prompt = f"""Paper: {title} ({paper.get('year', 'Unknown')})
Authors: {authors_str}
Venue: {paper.get('venue', 'Unknown')}
Citations: {paper.get('cited_by_count', 0)}

Abstract:
{abstract if abstract else 'No abstract available'}

Extract structured information based on this metadata."""

        processor.add_request(
            custom_id=custom_id,
            prompt=user_prompt,
            model=ModelTier.HAIKU,
            max_tokens=1024,
            system=METADATA_SUMMARY_EXTRACTION_SYSTEM,
            tools=[extraction_tool],
            tool_choice={"type": "tool", "name": "extract_metadata"},
        )

    logger.info(f"Submitting batch of {len(papers)} papers for metadata extraction")
    results = await processor.execute_batch()

    summaries = {}
    for custom_id, paper in paper_index.items():
        result = results.get(custom_id)
        doi = paper.get("doi", "unknown")
        title = paper.get("title", "Unknown Title")
        abstract = paper.get("abstract", "")
        authors = [a.get("name", "") for a in paper.get("authors", [])]

        if result and result.success:
            try:
                # Tool use returns valid JSON - just parse it
                extracted = json.loads(result.content)

                # Use existing short_summary from document processing if available,
                # otherwise fall back to abstract
                short_summary = existing_short_summaries.get(doi) or (abstract[:500] if abstract else f"Study on {title}")

                # Generate a Zotero key for fallback papers (consistent with sync path)
                generated_zotero_key = doi.replace("/", "_").replace(".", "")[:20].upper()

                summaries[doi] = PaperSummary(
                    doi=doi,
                    title=title,
                    authors=authors,
                    year=paper.get("year", 0),
                    venue=paper.get("venue"),
                    short_summary=short_summary,
                    es_record_id=None,
                    zotero_key=generated_zotero_key,
                    key_findings=extracted.get("key_findings", []),
                    methodology=extracted.get("methodology", "Not available from abstract"),
                    limitations=extracted.get("limitations", []),
                    future_work=extracted.get("future_work", []),
                    themes=extracted.get("themes", []),
                    claims=[],
                    relevance_score=paper.get("relevance_score", 0.6),
                    processing_status="metadata_only",
                )
            except Exception as e:
                logger.warning(f"Failed to parse metadata extraction for {doi}: {e}")
        else:
            error_msg = result.error if result else "No result returned"
            logger.warning(f"Metadata extraction failed for {doi}: {error_msg}")

    logger.info(f"Extracted {len(summaries)} metadata summaries (batch)")
    return summaries
