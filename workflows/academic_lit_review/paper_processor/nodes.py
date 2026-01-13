"""LangGraph node functions for paper processing.

Uses unified structured output interface that auto-selects batch API for 5+ papers.
"""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from core.stores.zotero import ZoteroItemCreate, ZoteroTag
from langchain_tools.base import get_store_manager
from workflows.academic_lit_review.state import PaperMetadata, PaperSummary
from workflows.shared.llm_utils import ModelTier
from workflows.shared.llm_utils.structured import (
    get_structured_output,
    StructuredRequest,
)


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
    Uses unified structured output interface that auto-selects batch API for 5+ papers.
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
        # Fallback to metadata-only extraction with real Zotero records
        # This ensures papers can still be cited even if full-text processing failed
        failed_dois = [p.get("doi", "unknown") for p in papers_needing_fallback]
        logger.warning(
            f"Document processing failed for {len(papers_needing_fallback)} papers. "
            f"Using metadata-only extraction with Zotero stubs. "
            f"DOIs: {failed_dois[:5]}{'...' if len(failed_dois) > 5 else ''}"
        )

        # Create real Zotero records for metadata-only papers
        paper_zotero_keys = await _create_zotero_stubs_for_papers(papers_needing_fallback)

        # Extract metadata summaries using the real Zotero keys
        metadata_summaries = await _extract_metadata_summaries_batched(
            papers=papers_needing_fallback,
            existing_short_summaries=existing_short_summaries,
            zotero_keys=paper_zotero_keys,
        )

        summaries.update(metadata_summaries)
        zotero_keys.update(paper_zotero_keys)

        logger.info(
            f"Metadata-only fallback complete: {len(metadata_summaries)} summaries, "
            f"{len(paper_zotero_keys)} Zotero records created"
        )

    if not summaries:
        logger.warning("No summaries extracted (no processing results and metadata fallback failed)")

    return {
        "paper_summaries": summaries,
        "elasticsearch_ids": es_ids,
        "zotero_keys": zotero_keys,
    }


async def _create_zotero_stubs_for_papers(
    papers: list[PaperMetadata],
) -> dict[str, str]:
    """Create Zotero records for papers that failed full document processing.

    This ensures metadata-only papers have real Zotero keys that can be verified
    by Loop 4's citation validation.

    Args:
        papers: List of paper metadata to create Zotero records for

    Returns:
        Dict mapping DOI -> Zotero key for created records
    """
    store_manager = get_store_manager()
    zotero_keys = {}

    for paper in papers:
        doi = paper.get("doi")
        if not doi:
            continue

        title = paper.get("title", "Unknown Title")
        authors = paper.get("authors", [])
        year = paper.get("year")
        venue = paper.get("venue")
        abstract = paper.get("abstract", "")

        # Build author string for Zotero
        author_names = [a.get("name", "") for a in authors[:10]]

        # Create Zotero item with paper metadata
        tags = [
            ZoteroTag(tag="metadata-only", type=1),  # Auto-tag indicating source
            ZoteroTag(tag="academic-lit-review", type=1),
        ]

        fields = {
            "title": title,
            "DOI": doi,
        }
        if abstract:
            fields["abstractNote"] = abstract[:2000]  # Zotero limit
        if year:
            fields["date"] = str(year)
        if venue:
            fields["publicationTitle"] = venue

        try:
            zotero_item = ZoteroItemCreate(
                itemType="journalArticle",
                fields=fields,
                tags=tags,
                creators=[
                    {"creatorType": "author", "name": name}
                    for name in author_names
                    if name
                ],
            )

            zotero_key = await store_manager.zotero.add(zotero_item)
            zotero_keys[doi] = zotero_key
            logger.debug(f"Created Zotero stub for {doi}: {zotero_key}")

        except Exception as e:
            # Fall back to generated key if Zotero creation fails
            fallback_key = doi.replace("/", "_").replace(".", "")[:20].upper()
            zotero_keys[doi] = fallback_key
            logger.warning(
                f"Failed to create Zotero record for {doi}, using fallback key: {e}"
            )

    logger.info(f"Created {len(zotero_keys)} Zotero stubs for metadata-only papers")
    return zotero_keys


async def _extract_metadata_summaries_batched(
    papers: list[PaperMetadata],
    existing_short_summaries: dict[str, str] | None = None,
    zotero_keys: dict[str, str] | None = None,
) -> dict[str, PaperSummary]:
    """Extract metadata summaries using unified structured output interface.

    Args:
        papers: List of paper metadata to process
        existing_short_summaries: Optional mapping of DOI -> short_summary from
            document processing (preferred over generating from abstract)
        zotero_keys: Optional mapping of DOI -> Zotero key for papers
            (if not provided, generates synthetic keys from DOI)
    """
    existing_short_summaries = existing_short_summaries or {}
    zotero_keys = zotero_keys or {}

    requests = []
    paper_index = {}

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

        requests.append(StructuredRequest(
            id=custom_id,
            user_prompt=user_prompt,
        ))

    logger.info(f"Submitting batch of {len(papers)} papers for metadata extraction")

    batch_results = await get_structured_output(
        output_schema=MetadataSummarySchema,
        requests=requests,
        system_prompt=METADATA_SUMMARY_EXTRACTION_SYSTEM,
        tier=ModelTier.HAIKU,
        max_tokens=1024,
    )

    summaries = {}
    for custom_id, paper in paper_index.items():
        result = batch_results.results.get(custom_id)
        doi = paper.get("doi", "unknown")
        title = paper.get("title", "Unknown Title")
        abstract = paper.get("abstract", "")
        authors = [a.get("name", "") for a in paper.get("authors", [])]

        if result and result.success:
            try:
                extracted = result.value

                # Use existing short_summary from document processing if available,
                # otherwise fall back to abstract
                short_summary = existing_short_summaries.get(doi) or (abstract[:500] if abstract else f"Study on {title}")

                # Use provided Zotero key if available, otherwise generate from DOI
                paper_zotero_key = zotero_keys.get(doi) or doi.replace("/", "_").replace(".", "")[:20].upper()

                summaries[doi] = PaperSummary(
                    doi=doi,
                    title=title,
                    authors=authors,
                    year=paper.get("year", 0),
                    venue=paper.get("venue"),
                    short_summary=short_summary,
                    es_record_id=None,
                    zotero_key=paper_zotero_key,
                    key_findings=extracted.key_findings,
                    methodology=extracted.methodology,
                    limitations=extracted.limitations,
                    future_work=extracted.future_work,
                    themes=extracted.themes,
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
