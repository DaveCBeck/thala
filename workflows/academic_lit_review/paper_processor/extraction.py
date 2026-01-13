"""Paper summary extraction from full text and metadata.

Uses unified structured output interface that auto-selects batch API for 5+ papers.
"""

import asyncio
import logging
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from langchain_tools.base import get_store_manager
from core.stores.schema import StoreRecord, SourceType
from workflows.academic_lit_review.state import (
    PaperMetadata,
    PaperSummary,
)
from workflows.shared.llm_utils import ModelTier, extract_json_cached
from workflows.shared.llm_utils.structured import (
    get_structured_output,
    StructuredRequest,
)
from workflows.shared.text_utils import count_words

# Threshold for L0 content size before generating L2 (150k chars ≈ 37k tokens)
L0_SIZE_THRESHOLD_FOR_L2 = 150_000


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


async def _generate_l2_from_l0(
    store_manager,
    l0_record_id: UUID,
    l0_content: str,
    zotero_key: str | None = None,
) -> str | None:
    """Generate L2 (10:1 summary) from L0 content when L2 doesn't exist.

    This is a fallback for cached papers that have L0 but no L2.
    Uses chapter detection and summarization from document_processing.

    Args:
        store_manager: Store manager instance
        l0_record_id: UUID of the L0 record
        l0_content: The L0 markdown content
        zotero_key: Optional Zotero key for the record

    Returns:
        L2 content if successfully generated, None otherwise
    """
    from workflows.document_processing.state import ChapterInfo
    from workflows.shared.chunking_utils import create_fallback_chunks
    from workflows.document_processing.subgraphs.chapter_summarization.chunking import (
        chunk_large_content,
    )
    from workflows.document_processing.subgraphs.chapter_summarization.nodes import (
        _summarize_content_chunk,
    )

    try:
        word_count = count_words(l0_content)
        logger.info(
            f"Generating L2 from L0 ({len(l0_content)} chars, {word_count} words) "
            f"for record {l0_record_id}"
        )

        # Skip if document is too short (same threshold as document_processing)
        if word_count < 3000:
            logger.info(f"Document too short ({word_count} words), skipping L2 generation")
            return None

        # Create fallback chunks (simpler than full chapter detection for papers)
        # Papers typically don't have chapter structure, so use size-based chunking
        chunks = create_fallback_chunks(l0_content, word_count, ChapterInfo)

        if not chunks:
            logger.warning("No chunks created, cannot generate L2")
            return None

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            # ChapterInfo is a TypedDict, so use dict access
            chunk_content = l0_content[chunk["start_position"]:chunk["end_position"]]
            target_words = max(50, chunk["word_count"] // 10)  # 10:1 compression

            # Handle very large chunks by sub-chunking
            sub_chunks = chunk_large_content(chunk_content)

            if len(sub_chunks) == 1:
                summary = await _summarize_content_chunk(
                    content=chunk_content,
                    target_words=target_words,
                    chapter_context=f"Section {i + 1} of {len(chunks)}",
                )
            else:
                # Large chunk - summarize sub-chunks
                sub_target = max(50, target_words // len(sub_chunks))
                sub_summaries = []
                for j, sub_chunk in enumerate(sub_chunks):
                    sub_summary = await _summarize_content_chunk(
                        content=sub_chunk,
                        target_words=sub_target,
                        chapter_context=f"Section {i + 1} of {len(chunks)}",
                        chunk_num=j + 1,
                        total_chunks=len(sub_chunks),
                    )
                    sub_summaries.append(sub_summary)
                summary = "\n\n".join(sub_summaries)

            chunk_summaries.append(summary)

        # Combine summaries
        tenth_summary = "\n\n".join(chunk_summaries)
        logger.info(
            f"Generated L2 summary: {count_words(tenth_summary)} words "
            f"(from {word_count} words, {len(chunks)} chunks)"
        )

        # Save L2 to store
        l2_record_id = uuid4()
        embedding = await store_manager.embedding.embed_long(tenth_summary)

        l2_record = StoreRecord(
            id=l2_record_id,
            source_type=SourceType.INTERNAL,
            zotero_key=zotero_key,
            content=tenth_summary,
            compression_level=2,
            source_ids=[l0_record_id],
            metadata={
                "type": "tenth_summary",
                "word_count": count_words(tenth_summary),
                "generated_from": "extraction_fallback",
            },
            embedding=embedding,
            embedding_model=store_manager.embedding.model,
        )

        await store_manager.es_stores.store.add(l2_record)
        logger.info(f"Saved L2 record {l2_record_id} (source: {l0_record_id})")

        return tenth_summary

    except Exception as e:
        logger.error(f"Failed to generate L2 from L0: {e}")
        return None


async def _fetch_content_for_extraction(store_manager, es_record_id: str, doi: str) -> str | None:
    """Fetch content for extraction, preferring L2 (10:1 summary) over L0 (original).

    L2 is preferred because:
    - For books/long documents, L2 captures the entire content in compressed form
    - L0 would require truncation for long documents, losing most of the content
    - L2 fits comfortably in LLM context windows

    Falls back to L0 if L2 is not available (e.g., short papers that skip 10:1 summarization).
    If L0 is too large (>150k chars), generates L2 on-the-fly and saves it for future use.

    Note: es_record_id is the L0 UUID. For L2, we use get_by_source_id() which
    searches by source_ids field (L1/L2 records store L0 UUID in source_ids).
    """
    record_uuid = UUID(es_record_id)
    store = store_manager.es_stores.store

    # Try L2 first (10:1 summary) - better for long documents
    # Use get_by_source_id since es_record_id is the L0 UUID
    record = await store.get_by_source_id(record_uuid, compression_level=2)
    if record and record.content:
        logger.debug(f"Using L2 (10:1 summary) for {doi}")
        return record.content

    # Fall back to L0 (original) for papers without L2
    l0_record = await store.get(record_uuid, compression_level=0)
    if not l0_record or not l0_record.content:
        return None

    l0_content = l0_record.content

    # If L0 is too large, generate L2 on-the-fly
    if len(l0_content) > L0_SIZE_THRESHOLD_FOR_L2:
        logger.info(
            f"L0 content for {doi} is {len(l0_content)} chars (>{L0_SIZE_THRESHOLD_FOR_L2}), "
            f"generating L2 summary"
        )
        l2_content = await _generate_l2_from_l0(
            store_manager=store_manager,
            l0_record_id=record_uuid,
            l0_content=l0_content,
            zotero_key=l0_record.zotero_key,
        )
        if l2_content:
            return l2_content
        # If L2 generation failed, fall through to return L0
        logger.warning(f"L2 generation failed for {doi}, falling back to L0")

    logger.debug(f"Using L0 (original) for {doi}")
    return l0_content

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

    # Use SONNET_1M for large content (>400k chars ≈ >100k tokens)
    # This threshold accounts for ~100k tokens of overhead (system prompt, metadata, response)
    # ensuring we stay under 200k limit for standard context, or use 1M for larger
    tier = ModelTier.SONNET_1M if len(content) > 400_000 else ModelTier.HAIKU

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
        extracted = await extract_json_cached(
            text=user_prompt,
            system_instructions=METADATA_SUMMARY_EXTRACTION_SYSTEM,
            tier=ModelTier.HAIKU,
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
                logger.info(f"[{completed_count}/{total_to_extract}] Extracted summary: {title}")

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
        logger.info(f"Submitting batch of {len(haiku_requests)} papers (HAIKU)")
        haiku_results = await get_structured_output(
            output_schema=PaperSummarySchema,
            requests=haiku_requests,
            system_prompt=PAPER_SUMMARY_EXTRACTION_SYSTEM,
            tier=ModelTier.HAIKU,
            max_tokens=2048,
        )
        all_results.update(haiku_results.results)

    if sonnet_1m_requests:
        logger.info(f"Submitting batch of {len(sonnet_1m_requests)} papers (SONNET_1M)")
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

    logger.info(f"Extracted summaries for {len(summaries)} papers (batch), {len(failed_dois)} failed")
    return summaries, es_ids, zotero_keys, failed_dois
