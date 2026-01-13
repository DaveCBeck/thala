"""Main extraction logic for paper summaries."""

import asyncio
import logging
from typing import Any

from langchain_tools.base import get_store_manager
from workflows.research.academic_lit_review.state import (
    PaperMetadata,
    PaperSummary,
)
from workflows.shared.llm_utils import ModelTier
from workflows.shared.llm_utils.structured import (
    get_structured_output,
    StructuredRequest,
)

from .parsers import _fetch_content_for_extraction
from .prompts import (
    PAPER_SUMMARY_EXTRACTION_SYSTEM,
    METADATA_SUMMARY_EXTRACTION_SYSTEM,
)
from .types import PaperSummarySchema

logger = logging.getLogger(__name__)


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

    # Use SONNET_1M for large content (>400k chars ≈ >100k tokens)
    # This threshold accounts for ~100k tokens of overhead (system prompt, metadata, response)
    # ensuring we stay under 200k limit for standard context, or use 1M for larger
    tier = ModelTier.SONNET_1M if len(content) > 400_000 else ModelTier.HAIKU

    try:
        extracted = await get_structured_output(
            output_schema=PaperSummarySchema,
            user_prompt=user_prompt,
            system_prompt=PAPER_SUMMARY_EXTRACTION_SYSTEM,
            tier=tier,
            enable_prompt_cache=True,
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
            key_findings=extracted.key_findings,
            methodology=extracted.methodology,
            limitations=extracted.limitations,
            future_work=extracted.future_work,
            themes=extracted.themes,
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
    existing_short_summary: str | None = None,
) -> PaperSummary:
    """Extract summary from paper metadata when full text is unavailable.

    Uses title, abstract, and other metadata to generate a summary.
    This is a fallback when document processing fails.

    Args:
        paper: Paper metadata including abstract
        existing_short_summary: Optional short_summary from document processing
            (preferred over generating from abstract if available)

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

    # Generate a Zotero key for fallback papers (consistent with sync path)
    generated_zotero_key = doi.replace("/", "_").replace(".", "")[:20].upper()

    try:
        extracted = await get_structured_output(
            output_schema=PaperSummarySchema,
            user_prompt=user_prompt,
            system_prompt=METADATA_SUMMARY_EXTRACTION_SYSTEM,
            tier=ModelTier.HAIKU,
            enable_prompt_cache=True,
        )

        # Use existing short_summary from document processing if available,
        # otherwise fall back to abstract
        short_summary = existing_short_summary or (abstract[:500] if abstract else f"Study on {title}")

        return PaperSummary(
            doi=doi,
            title=title,
            authors=authors,
            year=paper.get("year", 0),
            venue=paper.get("venue"),
            short_summary=short_summary,
            es_record_id=None,
            zotero_key=generated_zotero_key,
            key_findings=extracted.key_findings,
            methodology=extracted.methodology or "Not available from abstract",
            limitations=extracted.limitations,
            future_work=extracted.future_work,
            themes=extracted.themes,
            claims=[],
            relevance_score=paper.get("relevance_score", 0.6),
            processing_status="metadata_only",
        )

    except Exception as e:
        logger.warning(f"Failed to extract metadata summary for {doi}: {e}")
        # Use existing short_summary from document processing if available
        fallback_summary = existing_short_summary or (abstract[:500] if abstract else title)
        return PaperSummary(
            doi=doi,
            title=title,
            authors=authors,
            year=paper.get("year", 0),
            venue=paper.get("venue"),
            short_summary=fallback_summary,
            es_record_id=None,
            zotero_key=generated_zotero_key,
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
    use_batch_api: bool = True,
) -> tuple[dict[str, PaperSummary], dict[str, str], dict[str, str], set[str]]:
    """Extract structured summaries for all processed papers.

    Uses unified structured output interface that auto-selects batch API for 5+ papers.

    Args:
        processing_results: DOI -> processing result mapping
        papers_by_doi: DOI -> PaperMetadata mapping
        use_batch_api: Set False for rapid iteration (skips batch API)

    Returns:
        Tuple of (summaries, es_ids, zotero_keys, failed_dois)
        failed_dois contains DOIs that were in processing_results but failed extraction
    """
    store_manager = get_store_manager()

    if not processing_results:
        return {}, {}, {}, set()

    # Use batch API for 5+ papers when enabled
    if use_batch_api and len(processing_results) >= 5:
        return await _extract_all_summaries_batched(
            processing_results, papers_by_doi, store_manager
        )

    # Fall back to concurrent calls for small batches
    summaries = {}
    es_ids = {}
    zotero_keys = {}
    failed_dois = set()
    semaphore = asyncio.Semaphore(3)
    completed_count = 0
    total_to_extract = len(processing_results)

    async def extract_single_summary(doi: str, result: dict) -> tuple[str, Any, str, str, str]:
        """Returns (doi, summary, es_record_id, zotero_key, failure_reason)."""
        nonlocal completed_count
        async with semaphore:
            # Skip papers that were removed (e.g., by language verification)
            paper = papers_by_doi.get(doi)
            if paper is None:
                logger.debug(f"Skipping {doi}: not in papers_by_doi (likely filtered out)")
                return doi, None, None, None, "filtered_out"
            es_record_id = result.get("es_record_id")
            zotero_key = result.get("zotero_key")
            short_summary = result.get("short_summary", "")

            if not es_record_id:
                logger.warning(f"No ES record ID for {doi}, will need fallback")
                return doi, None, None, None, "no_es_record_id"

            try:
                content = await _fetch_content_for_extraction(
                    store_manager, es_record_id, doi
                )
                if not content:
                    logger.warning(f"Could not fetch content for {doi}, will need fallback")
                    return doi, None, None, None, "content_fetch_failed"

                summary = await extract_paper_summary(
                    content=content,
                    paper=paper,
                    short_summary=short_summary,
                    es_record_id=es_record_id,
                    zotero_key=zotero_key,
                )

                completed_count += 1
                title = paper.get("title", "Unknown")[:50]
                logger.debug(f"[{completed_count}/{total_to_extract}] Extracted summary: {title}")

                return doi, summary, es_record_id, zotero_key, None

            except Exception as e:
                logger.error(f"Failed to extract summary for {doi}: {e}")
                return doi, None, None, None, f"extraction_error: {e}"

    tasks = [extract_single_summary(doi, result) for doi, result in processing_results.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Summary extraction task failed: {result}")
            continue
        doi, summary, es_record_id, zotero_key, failure_reason = result
        if summary:
            summaries[doi] = summary
            es_ids[doi] = es_record_id
            zotero_keys[doi] = zotero_key
        elif failure_reason:
            failed_dois.add(doi)

    logger.info(f"Extracted summaries for {len(summaries)} papers, {len(failed_dois)} failed")
    return summaries, es_ids, zotero_keys, failed_dois


async def _extract_all_summaries_batched(
    processing_results: dict[str, dict],
    papers_by_doi: dict[str, PaperMetadata],
    store_manager,
) -> tuple[dict[str, PaperSummary], dict[str, str], dict[str, str], set[str]]:
    """Extract summaries using unified structured output interface.

    Returns:
        Tuple of (summaries, es_ids, zotero_keys, failed_dois)
    """
    # First, fetch all L0 content (this is I/O, not LLM calls)
    paper_data = {}  # doi -> {paper, content, es_record_id, zotero_key, short_summary}
    failed_dois = set()

    for doi, result in processing_results.items():
        # Skip papers that were removed (e.g., by language verification)
        paper = papers_by_doi.get(doi)
        if paper is None:
            logger.debug(f"Skipping {doi}: not in papers_by_doi (likely filtered out)")
            continue
        es_record_id = result.get("es_record_id")
        zotero_key = result.get("zotero_key")
        short_summary = result.get("short_summary", "")

        if not es_record_id:
            logger.warning(f"No ES record ID for {doi}, will need fallback")
            failed_dois.add(doi)
            continue

        try:
            content = await _fetch_content_for_extraction(
                store_manager, es_record_id, doi
            )
            if not content:
                logger.warning(f"Could not fetch content for {doi}, will need fallback")
                failed_dois.add(doi)
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
            failed_dois.add(doi)

    if not paper_data:
        return {}, {}, {}, failed_dois

    # Build batch requests, separated by tier
    haiku_requests = []
    sonnet_1m_requests = []
    doi_to_data = {}

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

        request = StructuredRequest(
            id=doi,
            user_prompt=user_prompt,
        )

        # Use SONNET_1M for large content (>400k chars ≈ >100k tokens)
        if len(content) > 400_000:
            sonnet_1m_requests.append(request)
            doi_to_data[doi] = (data, ModelTier.SONNET_1M)
        else:
            haiku_requests.append(request)
            doi_to_data[doi] = (data, ModelTier.HAIKU)

    # Run batches for each tier
    all_results = {}

    if haiku_requests:
        logger.debug(f"Submitting batch of {len(haiku_requests)} papers (HAIKU)")
        haiku_results = await get_structured_output(
            output_schema=PaperSummarySchema,
            requests=haiku_requests,
            system_prompt=PAPER_SUMMARY_EXTRACTION_SYSTEM,
            tier=ModelTier.HAIKU,
            max_tokens=2048,
        )
        all_results.update(haiku_results.results)

    if sonnet_1m_requests:
        logger.debug(f"Submitting batch of {len(sonnet_1m_requests)} papers (SONNET_1M)")
        sonnet_results = await get_structured_output(
            output_schema=PaperSummarySchema,
            requests=sonnet_1m_requests,
            system_prompt=PAPER_SUMMARY_EXTRACTION_SYSTEM,
            tier=ModelTier.SONNET_1M,
            max_tokens=2048,
        )
        all_results.update(sonnet_results.results)

    summaries = {}
    es_ids = {}
    zotero_keys = {}

    for doi in paper_data.keys():
        data, tier = doi_to_data[doi]
        paper = data["paper"]
        result = all_results.get(doi)

        if result and result.success:
            try:
                extracted = result.value

                summaries[doi] = PaperSummary(
                    doi=paper.get("doi"),
                    title=paper.get("title", "Unknown"),
                    authors=[a.get("name", "") for a in paper.get("authors", [])],
                    year=paper.get("year", 0),
                    venue=paper.get("venue"),
                    short_summary=data["short_summary"],
                    es_record_id=data["es_record_id"],
                    zotero_key=data["zotero_key"],
                    key_findings=extracted.key_findings,
                    methodology=extracted.methodology,
                    limitations=extracted.limitations,
                    future_work=extracted.future_work,
                    themes=extracted.themes,
                    claims=[],
                    relevance_score=0.7,
                    processing_status="success",
                )
                es_ids[doi] = data["es_record_id"]
                zotero_keys[doi] = data["zotero_key"]

            except Exception as e:
                logger.warning(f"Failed to parse extraction result for {doi}: {e}")
                failed_dois.add(doi)
        else:
            error_msg = result.error if result else "No result returned"
            logger.warning(f"Summary extraction failed for {doi}: {error_msg}")
            failed_dois.add(doi)

    logger.info(f"Extracted {len(summaries)} summaries (batch), {len(failed_dois)} failed")
    return summaries, es_ids, zotero_keys, failed_dois
