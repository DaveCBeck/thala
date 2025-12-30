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
# Each paper goes through acquire → process as a unit, which naturally rate-limits
# retrieval requests since document processing (LLM calls) takes significant time.
# Low concurrency (2) prevents overwhelming external retrieval sources.
MAX_PAPER_PIPELINE_CONCURRENT = 2
ACQUISITION_TIMEOUT = 120.0
RETRY_DELAY = 5.0
ACQUISITION_DELAY = 2.0  # Delay between acquisition requests to be polite to sources

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


async def acquire_and_process_single_paper(
    paper: PaperMetadata,
    client: RetrieveAcademicClient,
    output_dir: Path,
    paper_index: int,
    total_papers: int,
) -> dict[str, Any]:
    """Acquire full text and process a single paper as one unit.

    This combines acquisition and processing to naturally rate-limit
    retrieval requests (processing takes time, giving the source a break).

    Args:
        paper: Paper metadata
        client: Retrieve academic client
        output_dir: Directory to save downloaded files
        paper_index: Current paper number (for logging)
        total_papers: Total papers being processed

    Returns:
        Dict with acquisition and processing results
    """
    doi = paper.get("doi")
    title = paper.get("title", "Unknown")[:50]

    logger.info(f"[{paper_index}/{total_papers}] Processing: {title}...")

    result = {
        "doi": doi,
        "acquired": False,
        "local_path": None,
        "processing_result": None,
        "processing_success": False,
    }

    # Step 1: Acquire full text
    local_path = await acquire_full_text(paper, client, output_dir)

    if not local_path:
        logger.warning(f"[{paper_index}/{total_papers}] Acquisition failed for {doi}")
        return result

    result["acquired"] = True
    result["local_path"] = local_path

    # Step 2: Process document (this takes time, naturally rate-limiting)
    processing_result = await process_single_document(doi, local_path, paper)
    result["processing_result"] = processing_result
    result["processing_success"] = processing_result.get("success", False)

    if result["processing_success"]:
        logger.info(f"[{paper_index}/{total_papers}] Completed: {title}")
    else:
        logger.warning(f"[{paper_index}/{total_papers}] Processing failed for {doi}")

    return result


async def run_paper_pipeline(
    papers: list[PaperMetadata],
    max_concurrent: int = MAX_PAPER_PIPELINE_CONCURRENT,
) -> tuple[dict[str, str], dict[str, dict], list[str], list[str]]:
    """Run acquire→process pipeline for all papers with controlled concurrency.

    Uses a unified pipeline where each paper goes through acquisition and
    processing as one unit. This naturally rate-limits retrieval because
    document processing takes significant time.

    Args:
        papers: Papers to process
        max_concurrent: Maximum concurrent paper pipelines (default: 2)

    Returns:
        Tuple of (acquired, processing_results, acquisition_failed, processing_failed)
    """
    # Check if retrieve-academic service is available
    async with RetrieveAcademicClient() as client:
        if not await client.health_check():
            logger.warning("Retrieve-academic service unavailable, skipping full-text acquisition")
            return {}, {}, [p.get("doi") for p in papers], []

        # Create output directory
        output_dir = Path("/tmp/thala_papers")
        output_dir.mkdir(parents=True, exist_ok=True)

        acquired = {}
        processing_results = {}
        acquisition_failed = []
        processing_failed = []

        semaphore = asyncio.Semaphore(max_concurrent)
        total_papers = len(papers)

        async def process_with_limit(paper: PaperMetadata, index: int) -> dict:
            async with semaphore:
                # Add small delay between acquisitions to avoid rate limiting
                if index > 0:
                    await asyncio.sleep(ACQUISITION_DELAY)
                return await acquire_and_process_single_paper(
                    paper, client, output_dir, index + 1, total_papers
                )

        # Process papers with limited concurrency
        tasks = [process_with_limit(paper, i) for i, paper in enumerate(papers)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Paper pipeline task failed: {result}")
                continue

            doi = result.get("doi")
            if result.get("acquired"):
                acquired[doi] = result.get("local_path")

                if result.get("processing_success"):
                    processing_results[doi] = result.get("processing_result")
                else:
                    processing_failed.append(doi)
            else:
                acquisition_failed.append(doi)

        logger.info(
            f"Paper pipeline complete: {len(acquired)} acquired, "
            f"{len(processing_results)} processed, "
            f"{len(acquisition_failed)} acquisition failed, "
            f"{len(processing_failed)} processing failed"
        )
        return acquired, processing_results, acquisition_failed, processing_failed


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


METADATA_SUMMARY_EXTRACTION_SYSTEM = """Analyze this academic paper's metadata and abstract to extract structured information.

Extract the following in JSON format:
{
  "key_findings": ["2-3 inferred findings based on the abstract"],
  "methodology": "Inferred methodology from the abstract (1-2 sentences)",
  "limitations": [],
  "future_work": [],
  "themes": ["3-5 topic tags based on title and abstract"]
}

Note: This is based only on metadata/abstract, not full text. Be conservative and extract only what is clearly stated."""


async def extract_summary_from_metadata(
    paper: PaperMetadata,
) -> PaperSummary:
    """Extract summary from paper metadata when full text is unavailable.

    Uses title, abstract, and other metadata to generate a summary.
    This is a fallback when document processing fails.

    Args:
        paper: Paper metadata including abstract

    Returns:
        PaperSummary with fields populated from metadata
    """
    doi = paper.get("doi", "unknown")
    title = paper.get("title", "Unknown Title")
    abstract = paper.get("abstract", "")

    # Format authors
    authors = [a.get("name", "") for a in paper.get("authors", [])]
    authors_str = ", ".join(authors[:5])
    if len(authors) > 5:
        authors_str += " et al."

    # Create prompt from metadata
    user_prompt = f"""Paper: {title} ({paper.get('year', 'Unknown')})
Authors: {authors_str}
Venue: {paper.get('venue', 'Unknown')}
Citations: {paper.get('cited_by_count', 0)}

Abstract:
{abstract if abstract else 'No abstract available'}

Extract structured information based on this metadata."""

    try:
        extracted = await extract_json_cached(
            text=user_prompt,
            system_instructions=METADATA_SUMMARY_EXTRACTION_SYSTEM,
            tier=ModelTier.HAIKU,
        )

        # Use abstract as short summary if available
        short_summary = abstract[:500] if abstract else f"Study on {title}"

        return PaperSummary(
            doi=doi,
            title=title,
            authors=authors,
            year=paper.get("year", 0),
            venue=paper.get("venue"),
            short_summary=short_summary,
            es_record_id=None,  # No ES record for metadata-only
            zotero_key=None,  # No Zotero item for metadata-only
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
        logger.warning(f"Failed to extract metadata summary for {doi}: {e}")
        # Return minimal summary
        return PaperSummary(
            doi=doi,
            title=title,
            authors=authors,
            year=paper.get("year", 0),
            venue=paper.get("venue"),
            short_summary=abstract[:500] if abstract else title,
            es_record_id=None,
            zotero_key=None,
            key_findings=[],
            methodology="Not available",
            limitations=[],
            future_work=[],
            themes=[],
            claims=[],
            relevance_score=paper.get("relevance_score", 0.5),
            processing_status="metadata_minimal",
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

    # Run unified acquire→process pipeline
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

    # Build DOI -> PaperMetadata mapping
    papers_by_doi = {p.get("doi"): p for p in papers}

    summaries = {}
    es_ids = {}
    zotero_keys = {}

    # Extract from successfully processed documents
    if processing_results:
        full_text_summaries, full_text_es_ids, full_text_zotero_keys = await extract_all_summaries(
            processing_results=processing_results,
            papers_by_doi=papers_by_doi,
        )
        summaries.update(full_text_summaries)
        es_ids.update(full_text_es_ids)
        zotero_keys.update(full_text_zotero_keys)

    # Fallback: Extract from metadata for papers without full-text processing
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

        # Process in batches to avoid overwhelming the LLM
        semaphore = asyncio.Semaphore(10)

        async def extract_with_limit(paper: PaperMetadata) -> tuple[str, PaperSummary]:
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
                # Generate simple zotero key for metadata-only papers
                zotero_keys[doi] = doi.replace("/", "_").replace(".", "")[:20].upper()

        logger.info(f"Metadata fallback extracted {len(summaries) - len(processing_results)} additional summaries")

    if not summaries:
        logger.warning("No summaries extracted (no processing results and metadata fallback failed)")

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
        START -> acquire_and_process -> extract_summaries -> END

    The acquire_and_process node uses a unified pipeline where each paper
    goes through acquisition and processing as one unit. This naturally
    rate-limits retrieval requests since processing takes time.
    """
    builder = StateGraph(PaperProcessingState)

    # Add nodes
    builder.add_node("acquire_and_process", acquire_and_process_papers_node)
    builder.add_node("extract_summaries", extract_summaries_node)

    # Add edges
    builder.add_edge(START, "acquire_and_process")
    builder.add_edge("acquire_and_process", "extract_summaries")
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
