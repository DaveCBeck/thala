"""Paper summary extraction from full text and metadata.

Uses Anthropic Batch API for 50% cost reduction when processing 5+ papers.
"""

import asyncio
import json
import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from langchain_tools.base import get_store_manager
from workflows.research.subgraphs.academic_lit_review.state import (
    PaperMetadata,
    PaperSummary,
)
from workflows.shared.llm_utils import ModelTier, extract_json_cached
from workflows.shared.batch_processor import BatchProcessor


class PaperSummarySchema(BaseModel):
    """Schema for full-text paper summary extraction."""

    key_findings: list[str] = Field(
        default_factory=list,
        description="3-5 specific findings from the paper",
    )
    methodology: str = Field(
        default="Not specified",
        description="Brief research method description (1-2 sentences)",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Stated limitations from the paper",
    )
    future_work: list[str] = Field(
        default_factory=list,
        description="Suggested future research directions",
    )
    themes: list[str] = Field(
        default_factory=list,
        description="3-5 topic tags for clustering",
    )

logger = logging.getLogger(__name__)


async def _fetch_content_for_extraction(store_manager, es_record_id: str, doi: str) -> str | None:
    """Fetch content for extraction, preferring L2 (10:1 summary) over L0 (original).

    L2 is preferred because:
    - For books/long documents, L2 captures the entire content in compressed form
    - L0 would require truncation for long documents, losing most of the content
    - L2 fits comfortably in LLM context windows

    Falls back to L0 if L2 is not available (e.g., short papers that skip 10:1 summarization).
    """
    record_uuid = UUID(es_record_id)

    # Try L2 first (10:1 summary) - better for long documents
    record = await store_manager.es_stores.store.get(record_uuid, compression_level=2)
    if record and record.content:
        logger.debug(f"Using L2 (10:1 summary) for {doi}")
        return record.content

    # Fall back to L0 (original) for papers without L2
    record = await store_manager.es_stores.store.get(record_uuid, compression_level=0)
    if record and record.content:
        logger.debug(f"Using L0 (original) for {doi}")
        return record.content

    return None

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


async def extract_paper_summary(
    content: str,
    paper: PaperMetadata,
    short_summary: str,
    es_record_id: str,
    zotero_key: str,
) -> PaperSummary:
    """Extract structured summary from paper content.

    Args:
        content: Paper content (L2 10:1 summary preferred, L0 fallback)
        paper: Paper metadata
        short_summary: 100-word summary from document_processing
        es_record_id: Elasticsearch record ID
        zotero_key: Zotero item key

    Returns:
        PaperSummary with extracted fields
    """
    authors_str = ", ".join(
        a.get("name", "") for a in paper.get("authors", [])[:5]
    )
    if len(paper.get("authors", [])) > 5:
        authors_str += " et al."

    user_prompt = f"""Paper: {paper.get('title', 'Unknown')} ({paper.get('year', 'Unknown')})
Authors: {authors_str}
Venue: {paper.get('venue', 'Unknown')}

Content:
{content}

Extract structured information from this paper."""

    # Use SONNET_1M for large content (>600k chars ≈ >150k tokens)
    # Most L2 summaries are <50k chars; only L0 fallbacks for books might be larger
    tier = ModelTier.SONNET_1M if len(content) > 600_000 else ModelTier.HAIKU

    try:
        extracted = await extract_json_cached(
            text=user_prompt,
            system_instructions=PAPER_SUMMARY_EXTRACTION_SYSTEM,
            tier=tier,
        )

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
            claims=[],
            relevance_score=0.7,
            processing_status="success",
        )

    except Exception as e:
        logger.warning(f"Failed to extract summary for {paper.get('doi')}: {e}")
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

    try:
        extracted = await extract_json_cached(
            text=user_prompt,
            system_instructions=METADATA_SUMMARY_EXTRACTION_SYSTEM,
            tier=ModelTier.HAIKU,
        )

        short_summary = abstract[:500] if abstract else f"Study on {title}"

        return PaperSummary(
            doi=doi,
            title=title,
            authors=authors,
            year=paper.get("year", 0),
            venue=paper.get("venue"),
            short_summary=short_summary,
            es_record_id=None,
            zotero_key=None,
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

    Uses Anthropic Batch API for 50% cost reduction when processing 5+ papers.

    Args:
        processing_results: DOI -> processing result mapping
        papers_by_doi: DOI -> PaperMetadata mapping

    Returns:
        Tuple of (summaries, es_ids, zotero_keys)
    """
    store_manager = get_store_manager()

    if not processing_results:
        return {}, {}, {}

    # Use batch API for 5+ papers
    if len(processing_results) >= 5:
        return await _extract_all_summaries_batched(
            processing_results, papers_by_doi, store_manager
        )

    # Fall back to concurrent calls for small batches
    summaries = {}
    es_ids = {}
    zotero_keys = {}
    semaphore = asyncio.Semaphore(3)
    completed_count = 0
    total_to_extract = len(processing_results)

    async def extract_single_summary(doi: str, result: dict) -> tuple[str, Any, str, str]:
        nonlocal completed_count
        async with semaphore:
            paper = papers_by_doi[doi]
            es_record_id = result.get("es_record_id")
            zotero_key = result.get("zotero_key")
            short_summary = result.get("short_summary", "")

            if not es_record_id:
                logger.warning(f"No ES record ID for {doi}, skipping summary extraction")
                return doi, None, None, None

            try:
                content = await _fetch_content_for_extraction(
                    store_manager, es_record_id, doi
                )
                if not content:
                    logger.warning(f"Could not fetch content for {doi}")
                    return doi, None, None, None

                summary = await extract_paper_summary(
                    content=content,
                    paper=paper,
                    short_summary=short_summary,
                    es_record_id=es_record_id,
                    zotero_key=zotero_key,
                )

                completed_count += 1
                title = paper.get("title", "Unknown")[:50]
                logger.info(f"[{completed_count}/{total_to_extract}] Extracted summary: {title}")

                return doi, summary, es_record_id, zotero_key

            except Exception as e:
                logger.error(f"Failed to extract summary for {doi}: {e}")
                return doi, None, None, None

    tasks = [extract_single_summary(doi, result) for doi, result in processing_results.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Summary extraction task failed: {result}")
            continue
        doi, summary, es_record_id, zotero_key = result
        if summary:
            summaries[doi] = summary
            es_ids[doi] = es_record_id
            zotero_keys[doi] = zotero_key

    logger.info(f"Extracted summaries for {len(summaries)} papers")
    return summaries, es_ids, zotero_keys


async def _extract_all_summaries_batched(
    processing_results: dict[str, dict],
    papers_by_doi: dict[str, PaperMetadata],
    store_manager,
) -> tuple[dict[str, PaperSummary], dict[str, str], dict[str, str]]:
    """Extract summaries using Anthropic Batch API for 50% cost reduction.

    Uses tool calling to guarantee valid JSON responses.
    """
    # First, fetch all L0 content (this is I/O, not LLM calls)
    paper_data = {}  # doi -> {paper, content, es_record_id, zotero_key, short_summary}

    for doi, result in processing_results.items():
        paper = papers_by_doi[doi]
        es_record_id = result.get("es_record_id")
        zotero_key = result.get("zotero_key")
        short_summary = result.get("short_summary", "")

        if not es_record_id:
            logger.warning(f"No ES record ID for {doi}, skipping")
            continue

        try:
            content = await _fetch_content_for_extraction(
                store_manager, es_record_id, doi
            )
            if not content:
                logger.warning(f"Could not fetch content for {doi}")
                continue

            paper_data[doi] = {
                "paper": paper,
                "content": content,
                "es_record_id": es_record_id,
                "zotero_key": zotero_key,
                "short_summary": short_summary,
            }
        except Exception as e:
            logger.error(f"Failed to fetch content for {doi}: {e}")

    if not paper_data:
        return {}, {}, {}

    # Build batch requests with tool calling for guaranteed JSON
    processor = BatchProcessor(poll_interval=30)

    extraction_tool = {
        "name": "extract_summary",
        "description": "Extract structured summary from an academic paper",
        "input_schema": PaperSummarySchema.model_json_schema(),
    }

    for doi, data in paper_data.items():
        paper = data["paper"]
        content = data["content"]

        authors_str = ", ".join(
            a.get("name", "") for a in paper.get("authors", [])[:5]
        )
        if len(paper.get("authors", [])) > 5:
            authors_str += " et al."

        user_prompt = f"""Paper: {paper.get('title', 'Unknown')} ({paper.get('year', 'Unknown')})
Authors: {authors_str}
Venue: {paper.get('venue', 'Unknown')}

Content:
{content}

Extract structured information from this paper."""

        # Use SONNET_1M for large content (>600k chars ≈ >150k tokens)
        tier = ModelTier.SONNET_1M if len(content) > 600_000 else ModelTier.HAIKU

        processor.add_request(
            custom_id=doi,
            prompt=user_prompt,
            model=tier,
            max_tokens=2048,
            system=PAPER_SUMMARY_EXTRACTION_SYSTEM,
            tools=[extraction_tool],
            tool_choice={"type": "tool", "name": "extract_summary"},
        )

    logger.info(f"Submitting batch of {len(paper_data)} papers for summary extraction")
    results = await processor.execute_batch()

    summaries = {}
    es_ids = {}
    zotero_keys = {}

    for doi, data in paper_data.items():
        paper = data["paper"]
        result = results.get(doi)

        if result and result.success:
            try:
                # Tool use returns valid JSON - just parse it
                extracted = json.loads(result.content)

                summaries[doi] = PaperSummary(
                    doi=paper.get("doi"),
                    title=paper.get("title", "Unknown"),
                    authors=[a.get("name", "") for a in paper.get("authors", [])],
                    year=paper.get("year", 0),
                    venue=paper.get("venue"),
                    short_summary=data["short_summary"],
                    es_record_id=data["es_record_id"],
                    zotero_key=data["zotero_key"],
                    key_findings=extracted.get("key_findings", []),
                    methodology=extracted.get("methodology", "Not specified"),
                    limitations=extracted.get("limitations", []),
                    future_work=extracted.get("future_work", []),
                    themes=extracted.get("themes", []),
                    claims=[],
                    relevance_score=0.7,
                    processing_status="success",
                )
                es_ids[doi] = data["es_record_id"]
                zotero_keys[doi] = data["zotero_key"]

            except Exception as e:
                logger.warning(f"Failed to parse extraction result for {doi}: {e}")
        else:
            error_msg = result.error if result else "No result returned"
            logger.warning(f"Summary extraction failed for {doi}: {error_msg}")

    logger.info(f"Extracted summaries for {len(summaries)} papers (batch)")
    return summaries, es_ids, zotero_keys
