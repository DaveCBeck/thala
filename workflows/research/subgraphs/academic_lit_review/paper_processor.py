"""Paper processor subgraph for academic literature review.

Processes papers from diffusion through:
1. Full-text acquisition via retrieve-academic service
2. PDF→Markdown processing via document_processing workflow
3. Structured PaperSummary extraction for clustering

Flow:
    START -> acquire_papers -> process_documents -> extract_summaries -> END
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from core.stores.retrieve_academic import RetrieveAcademicClient
from langchain_tools.base import get_store_manager
from workflows.document_processing.graph import process_document
from workflows.research.subgraphs.academic_lit_review.state import (
    LitReviewInput,
    PaperMetadata,
    PaperSummary,
    QualitySettings,
)
from workflows.shared.llm_utils import ModelTier, get_llm, extract_json_cached

logger = logging.getLogger(__name__)

# Constants
MAX_ACQUISITION_CONCURRENT = 5
MAX_PROCESSING_CONCURRENT = 3
ACQUISITION_TIMEOUT = 120.0
RETRY_DELAY = 5.0

# Paper summary extraction prompt
PAPER_SUMMARY_EXTRACTION_SYSTEM = """Analyze this academic paper and extract structured information.

Extract the following in JSON format:
{
  "key_findings": ["3-5 specific findings from the paper"],
  "methodology": "Brief research method description (1-2 sentences)",
  "limitations": ["Stated limitations from the paper"],
  "future_work": ["Suggested future research directions"],
  "themes": ["3-5 topic tags for clustering"]
}

Be specific and grounded in the paper content. Do not hallucinate information.
If a field is not present in the paper, use an empty list or brief note."""


# =============================================================================
# State Definition
# =============================================================================


class PaperProcessingState(TypedDict):
    """State for paper processing subgraph."""

    # Input
    input: LitReviewInput
    quality_settings: QualitySettings
    papers_to_process: list[PaperMetadata]

    # Acquisition tracking
    acquired_papers: dict[str, str]  # DOI -> local file path
    acquisition_failed: list[str]  # DOIs that failed acquisition

    # Processing tracking
    processing_results: dict[str, dict]  # DOI -> processing result
    processing_failed: list[str]  # DOIs that failed processing

    # Output
    paper_summaries: dict[str, PaperSummary]  # DOI -> extracted summary
    elasticsearch_ids: dict[str, str]  # DOI -> ES record ID
    zotero_keys: dict[str, str]  # DOI -> Zotero key


# =============================================================================
# Helper Functions
# =============================================================================


async def acquire_full_text(
    paper: PaperMetadata,
    client: RetrieveAcademicClient,
    output_dir: Path,
) -> Optional[str]:
    """Acquire full text for a single paper.

    Args:
        paper: Paper metadata
        client: Retrieve academic client
        output_dir: Directory to save downloaded files

    Returns:
        Local file path or None on failure
    """
    doi = paper.get("doi")
    title = paper.get("title", "Unknown")

    # First check if open access URL is available
    oa_url = paper.get("oa_url")
    if oa_url:
        logger.debug(f"Paper {doi} has OA URL: {oa_url}")
        # Could download directly, but retrieve-academic handles this too
        # Fall through to use retrieve-academic for consistency

    # Prepare author list
    authors = []
    for author in paper.get("authors", [])[:5]:
        name = author.get("name")
        if name:
            authors.append(name)

    # Generate local path
    safe_doi = doi.replace("/", "_").replace(":", "_")
    local_path = output_dir / f"{safe_doi}.pdf"

    try:
        saved_path, result = await client.retrieve_and_download(
            doi=doi,
            local_path=str(local_path),
            title=title,
            authors=authors,
            timeout=ACQUISITION_TIMEOUT,
        )
        logger.info(f"Acquired full text for {doi}: {saved_path}")
        return saved_path

    except Exception as e:
        logger.warning(f"Failed to acquire full text for {doi}: {e}")
        return None


async def acquire_all_papers(
    papers: list[PaperMetadata],
    max_concurrent: int = MAX_ACQUISITION_CONCURRENT,
) -> tuple[dict[str, str], list[str]]:
    """Acquire full text for all papers in parallel.

    Args:
        papers: Papers to acquire
        max_concurrent: Maximum concurrent acquisitions

    Returns:
        Tuple of (acquired_dict, failed_dois)
        acquired_dict: DOI -> local file path
        failed_dois: List of DOIs that failed
    """
    # Check if retrieve-academic service is available
    async with RetrieveAcademicClient() as client:
        if not await client.health_check():
            logger.warning("Retrieve-academic service unavailable, skipping full-text acquisition")
            return {}, [p.get("doi") for p in papers]

        # Create output directory
        output_dir = Path("/tmp/thala_papers")
        output_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max_concurrent)
        acquired = {}
        failed = []

        async def acquire_with_limit(paper: PaperMetadata) -> tuple[str, Optional[str]]:
            async with semaphore:
                doi = paper.get("doi")
                result = await acquire_full_text(paper, client, output_dir)
                return doi, result

        # Execute acquisitions in parallel
        tasks = [acquire_with_limit(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Acquisition task failed: {result}")
                continue

            doi, local_path = result
            if local_path:
                acquired[doi] = local_path
            else:
                failed.append(doi)

        logger.info(
            f"Full-text acquisition: {len(acquired)} acquired, {len(failed)} failed"
        )
        return acquired, failed


async def process_single_document(
    doi: str,
    local_path: str,
    paper: PaperMetadata,
) -> dict[str, Any]:
    """Process a single document through document_processing workflow.

    Args:
        doi: Paper DOI
        local_path: Local file path
        paper: Paper metadata

    Returns:
        Processing result with es_record_id, zotero_key, etc.
    """
    try:
        # Build extra metadata for Zotero
        extra_metadata = {
            "DOI": doi,
            "date": paper.get("publication_date", ""),
            "publicationTitle": paper.get("venue", ""),
            "abstractNote": paper.get("abstract", "")[:500] if paper.get("abstract") else "",
        }

        result = await process_document(
            source=local_path,
            title=paper.get("title", "Unknown"),
            item_type="journalArticle",
            quality="fast",  # Use fast processing for large batches
            extra_metadata=extra_metadata,
        )

        # Extract key fields
        return {
            "doi": doi,
            "success": result.get("current_status") != "failed",
            "es_record_id": result.get("store_record_id"),
            "zotero_key": result.get("zotero_key"),
            "short_summary": result.get("short_summary", ""),
            "errors": result.get("errors", []),
        }

    except Exception as e:
        logger.error(f"Failed to process document {doi}: {e}")
        return {
            "doi": doi,
            "success": False,
            "errors": [{"node": "process_document", "error": str(e)}],
        }


async def process_papers_batch(
    acquired: dict[str, str],
    papers_by_doi: dict[str, PaperMetadata],
    max_concurrent: int = MAX_PROCESSING_CONCURRENT,
) -> tuple[dict[str, dict], list[str]]:
    """Process acquired papers through document_processing workflow.

    Args:
        acquired: DOI -> local file path mapping
        papers_by_doi: DOI -> PaperMetadata mapping
        max_concurrent: Maximum concurrent processing

    Returns:
        Tuple of (processing_results, failed_dois)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    failed = []

    async def process_with_limit(doi: str, local_path: str) -> dict:
        async with semaphore:
            paper = papers_by_doi[doi]
            return await process_single_document(doi, local_path, paper)

    # Process all papers
    tasks = [process_with_limit(doi, path) for doi, path in acquired.items()]
    processing_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in processing_results:
        if isinstance(result, Exception):
            logger.error(f"Processing task failed: {result}")
            continue

        doi = result.get("doi")
        if result.get("success"):
            results[doi] = result
        else:
            failed.append(doi)
            logger.warning(f"Document processing failed for {doi}: {result.get('errors')}")

    logger.info(
        f"Document processing: {len(results)} succeeded, {len(failed)} failed"
    )
    return results, failed


async def extract_paper_summary(
    content: str,
    paper: PaperMetadata,
    short_summary: str,
    es_record_id: str,
    zotero_key: str,
) -> PaperSummary:
    """Extract structured summary from paper content.

    Args:
        content: Full paper content (L0 markdown)
        paper: Paper metadata
        short_summary: 100-word summary from document_processing
        es_record_id: Elasticsearch record ID
        zotero_key: Zotero item key

    Returns:
        PaperSummary with extracted fields
    """
    # Format paper metadata for prompt
    authors_str = ", ".join(
        a.get("name", "") for a in paper.get("authors", [])[:5]
    )
    if len(paper.get("authors", [])) > 5:
        authors_str += " et al."

    user_prompt = f"""Paper: {paper.get('title', 'Unknown')} ({paper.get('year', 'Unknown')})
Authors: {authors_str}
Venue: {paper.get('venue', 'Unknown')}

Content (first 40k chars):
{content[:40000]}

Extract structured information from this paper."""

    try:
        llm = get_llm(tier=ModelTier.HAIKU)  # Use Haiku for cost efficiency
        extracted = await extract_json_cached(
            text=user_prompt,
            system_instructions=PAPER_SUMMARY_EXTRACTION_SYSTEM,
            tier=ModelTier.HAIKU,
        )

        # Build PaperSummary
        return PaperSummary(
            doi=paper.get("doi"),
            title=paper.get("title", "Unknown"),
            authors=[a.get("name", "") for a in paper.get("authors", [])],
            year=paper.get("year", 0),
            venue=paper.get("venue"),
            short_summary=short_summary,
            es_record_id=es_record_id,
            zotero_key=zotero_key,
            key_findings=extracted.get("key_findings", []),
            methodology=extracted.get("methodology", "Not specified"),
            limitations=extracted.get("limitations", []),
            future_work=extracted.get("future_work", []),
            themes=extracted.get("themes", []),
            claims=[],  # Claims extraction could be added later
            relevance_score=0.7,  # Default, could be re-scored
            processing_status="success",
        )

    except Exception as e:
        logger.warning(f"Failed to extract summary for {paper.get('doi')}: {e}")
        # Return minimal summary on failure
        return PaperSummary(
            doi=paper.get("doi"),
            title=paper.get("title", "Unknown"),
            authors=[a.get("name", "") for a in paper.get("authors", [])],
            year=paper.get("year", 0),
            venue=paper.get("venue"),
            short_summary=short_summary,
            es_record_id=es_record_id,
            zotero_key=zotero_key,
            key_findings=[],
            methodology="Extraction failed",
            limitations=[],
            future_work=[],
            themes=[],
            claims=[],
            relevance_score=0.7,
            processing_status="partial",
        )


async def extract_all_summaries(
    processing_results: dict[str, dict],
    papers_by_doi: dict[str, PaperMetadata],
) -> tuple[dict[str, PaperSummary], dict[str, str], dict[str, str]]:
    """Extract structured summaries for all processed papers.

    Args:
        processing_results: DOI -> processing result mapping
        papers_by_doi: DOI -> PaperMetadata mapping

    Returns:
        Tuple of (summaries, es_ids, zotero_keys)
    """
    store_manager = get_store_manager()
    summaries = {}
    es_ids = {}
    zotero_keys = {}

    for doi, result in processing_results.items():
        paper = papers_by_doi[doi]
        es_record_id = result.get("es_record_id")
        zotero_key = result.get("zotero_key")
        short_summary = result.get("short_summary", "")

        if not es_record_id:
            logger.warning(f"No ES record ID for {doi}, skipping summary extraction")
            continue

        # Fetch L0 content from Elasticsearch
        try:
            record = await store_manager.es_stores.store.get(
                UUID(es_record_id),
                index="store_l0"
            )
            if not record:
                logger.warning(f"Could not fetch L0 content for {doi}")
                continue

            content = record.content

            # Extract summary
            summary = await extract_paper_summary(
                content=content,
                paper=paper,
                short_summary=short_summary,
                es_record_id=es_record_id,
                zotero_key=zotero_key,
            )

            summaries[doi] = summary
            es_ids[doi] = es_record_id
            zotero_keys[doi] = zotero_key

        except Exception as e:
            logger.error(f"Failed to extract summary for {doi}: {e}")
            continue

    logger.info(f"Extracted summaries for {len(summaries)} papers")
    return summaries, es_ids, zotero_keys


# =============================================================================
# Node Functions
# =============================================================================


async def acquire_papers_node(state: PaperProcessingState) -> dict[str, Any]:
    """Acquire full text for all papers."""
    papers = state.get("papers_to_process", [])

    if not papers:
        logger.warning("No papers to acquire")
        return {
            "acquired_papers": {},
            "acquisition_failed": [],
        }

    acquired, failed = await acquire_all_papers(papers)

    return {
        "acquired_papers": acquired,
        "acquisition_failed": failed,
    }


async def process_documents_node(state: PaperProcessingState) -> dict[str, Any]:
    """Process acquired documents through document_processing workflow."""
    acquired = state.get("acquired_papers", {})
    papers = state.get("papers_to_process", [])

    # Build DOI -> PaperMetadata mapping
    papers_by_doi = {p.get("doi"): p for p in papers}

    if not acquired:
        logger.warning("No acquired papers to process")
        return {
            "processing_results": {},
            "processing_failed": [],
        }

    # Use batch processing for large batches
    quality_settings = state.get("quality_settings", {})
    use_batch = quality_settings.get("use_batch_api", False) and len(acquired) >= 20

    if use_batch:
        logger.info(f"Using batch API for {len(acquired)} papers (50% cost savings)")
        # TODO: Implement batch API mode via workflows/shared/batch_processor.py
        # For now, fall back to concurrent processing
        use_batch = False

    results, failed = await process_papers_batch(
        acquired=acquired,
        papers_by_doi=papers_by_doi,
    )

    return {
        "processing_results": results,
        "processing_failed": failed,
    }


async def extract_summaries_node(state: PaperProcessingState) -> dict[str, Any]:
    """Extract structured summaries from processed papers."""
    processing_results = state.get("processing_results", {})
    papers = state.get("papers_to_process", [])

    # Build DOI -> PaperMetadata mapping
    papers_by_doi = {p.get("doi"): p for p in papers}

    if not processing_results:
        logger.warning("No processing results to extract summaries from")
        return {
            "paper_summaries": {},
            "elasticsearch_ids": {},
            "zotero_keys": {},
        }

    summaries, es_ids, zotero_keys = await extract_all_summaries(
        processing_results=processing_results,
        papers_by_doi=papers_by_doi,
    )

    return {
        "paper_summaries": summaries,
        "elasticsearch_ids": es_ids,
        "zotero_keys": zotero_keys,
    }


# =============================================================================
# Subgraph Definition
# =============================================================================


def create_paper_processing_subgraph() -> StateGraph:
    """Create the paper processing subgraph.

    Flow:
        START -> acquire_papers -> process_documents -> extract_summaries -> END
    """
    builder = StateGraph(PaperProcessingState)

    # Add nodes
    builder.add_node("acquire_papers", acquire_papers_node)
    builder.add_node("process_documents", process_documents_node)
    builder.add_node("extract_summaries", extract_summaries_node)

    # Add edges
    builder.add_edge(START, "acquire_papers")
    builder.add_edge("acquire_papers", "process_documents")
    builder.add_edge("process_documents", "extract_summaries")
    builder.add_edge("extract_summaries", END)

    return builder.compile()


# Export compiled subgraph
paper_processing_subgraph = create_paper_processing_subgraph()


# =============================================================================
# Convenience Function
# =============================================================================


async def run_paper_processing(
    papers: list[PaperMetadata],
    quality_settings: QualitySettings,
    topic: str,
) -> dict[str, Any]:
    """Run paper processing as standalone operation.

    Args:
        papers: Papers to process
        quality_settings: Quality tier settings
        topic: Research topic

    Returns:
        Dict with paper_summaries, elasticsearch_ids, zotero_keys
    """
    from workflows.research.subgraphs.academic_lit_review.state import LitReviewInput

    input_data = LitReviewInput(
        topic=topic,
        research_questions=[],
        quality="standard",
        date_range=None,
        include_books=False,
        focus_areas=None,
        exclude_terms=None,
        max_papers=None,
    )

    initial_state = PaperProcessingState(
        input=input_data,
        quality_settings=quality_settings,
        papers_to_process=papers,
        acquired_papers={},
        acquisition_failed=[],
        processing_results={},
        processing_failed=[],
        paper_summaries={},
        elasticsearch_ids={},
        zotero_keys={},
    )

    result = await paper_processing_subgraph.ainvoke(initial_state)
    return {
        "paper_summaries": result.get("paper_summaries", {}),
        "elasticsearch_ids": result.get("elasticsearch_ids", {}),
        "zotero_keys": result.get("zotero_keys", {}),
        "acquisition_failed": result.get("acquisition_failed", []),
        "processing_failed": result.get("processing_failed", []),
    }
