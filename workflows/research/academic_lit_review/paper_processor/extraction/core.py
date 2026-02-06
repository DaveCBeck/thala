"""Main extraction logic for paper summaries.

Routes through central LLM broker for unified cost/speed management.
"""

import logging

from langchain_tools.base import get_store_manager

from workflows.research.academic_lit_review.state import (
    PaperMetadata,
    PaperSummary,
)
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

from .parsers import _fetch_content_for_extraction
from .prompts import (
    METADATA_SUMMARY_EXTRACTION_SYSTEM,
    PAPER_SUMMARY_EXTRACTION_SYSTEM,
    format_paper_extraction_system,
)
from .types import PaperSummarySchema

logger = logging.getLogger(__name__)


async def extract_paper_summary(
    content: str,
    paper: PaperMetadata,
    short_summary: str,
    es_record_id: str,
    zotero_key: str,
    topic: str | None = None,
    research_questions: list[str] | None = None,
) -> PaperSummary:
    """Extract structured summary from paper content.

    Args:
        content: Paper content (L2 10:1 summary preferred, L0 fallback)
        paper: Paper metadata
        short_summary: 100-word summary from document_processing
        es_record_id: Elasticsearch record ID
        zotero_key: Zotero item key
        topic: Research topic for context-aware extraction
        research_questions: Research questions to focus extraction

    Returns:
        PaperSummary with extracted fields
    """
    authors_str = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:5])
    if len(paper.get("authors", [])) > 5:
        authors_str += " et al."

    user_prompt = f"""Paper: {paper.get("title", "Unknown")} ({paper.get("year", "Unknown")})
Authors: {authors_str}
Venue: {paper.get("venue", "Unknown")}

Content:
{content}

Extract structured information from this paper."""

    # Use research-context-aware prompt if topic provided, otherwise fallback
    if topic and research_questions:
        system_prompt = format_paper_extraction_system(topic, research_questions)
    else:
        system_prompt = PAPER_SUMMARY_EXTRACTION_SYSTEM

    # Use SONNET_1M for large content (>400k chars ≈ >100k tokens)
    # DeepSeek V3 has 128K context (~400k chars), so use Sonnet 1M for larger
    # DeepSeek V3 is ~5x cheaper than Haiku with comparable quality for extraction
    tier = ModelTier.SONNET_1M if len(content) > 400_000 else ModelTier.DEEPSEEK_V3

    try:
        extracted = await invoke(
            tier=tier,
            system=system_prompt,
            user=user_prompt,
            schema=PaperSummarySchema,
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

    user_prompt = f"""Paper: {title} ({paper.get("year", "Unknown")})
Authors: {authors_str}
Venue: {paper.get("venue", "Unknown")}
Citations: {paper.get("cited_by_count", 0)}

Abstract:
{abstract if abstract else "No abstract available"}

Extract structured information based on this metadata."""

    # Generate a Zotero key for fallback papers (consistent with sync path)
    generated_zotero_key = doi.replace("/", "_").replace(".", "")[:20].upper()

    try:
        extracted = await invoke(
            tier=ModelTier.DEEPSEEK_V3,
            system=METADATA_SUMMARY_EXTRACTION_SYSTEM,
            user=user_prompt,
            schema=PaperSummarySchema,
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
    topic: str | None = None,
    research_questions: list[str] | None = None,
) -> tuple[dict[str, PaperSummary], dict[str, str], dict[str, str], set[str]]:
    """Extract structured summaries for all processed papers.

    Routes through central LLM broker for unified cost/speed management.

    Args:
        processing_results: DOI -> processing result mapping
        papers_by_doi: DOI -> PaperMetadata mapping
        topic: Research topic for context-aware extraction
        research_questions: Research questions to focus extraction

    Returns:
        Tuple of (summaries, es_ids, zotero_keys, failed_dois)
        failed_dois contains DOIs that were in processing_results but failed extraction
    """
    store_manager = get_store_manager()

    if not processing_results:
        return {}, {}, {}, set()

    # Build system prompt with research context if available
    if topic and research_questions:
        system_prompt = format_paper_extraction_system(topic, research_questions)
    else:
        system_prompt = PAPER_SUMMARY_EXTRACTION_SYSTEM

    # First, fetch all content (this is I/O, not LLM calls)
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
            content = await _fetch_content_for_extraction(store_manager, es_record_id, doi)
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

    # Build batch prompts, separated by tier
    deepseek_prompts: list[str] = []
    deepseek_dois: list[str] = []
    sonnet_1m_prompts: list[str] = []
    sonnet_1m_dois: list[str] = []
    doi_to_data = {}

    for doi, data in paper_data.items():
        paper = data["paper"]
        content = data["content"]

        authors_str = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:5])
        if len(paper.get("authors", [])) > 5:
            authors_str += " et al."

        user_prompt = f"""Paper: {paper.get("title", "Unknown")} ({paper.get("year", "Unknown")})
Authors: {authors_str}
Venue: {paper.get("venue", "Unknown")}

Content:
{content}

Extract structured information from this paper."""

        # Use SONNET_1M for large content (>400k chars ≈ >100k tokens)
        if len(content) > 400_000:
            sonnet_1m_prompts.append(user_prompt)
            sonnet_1m_dois.append(doi)
            doi_to_data[doi] = (data, ModelTier.SONNET_1M)
        else:
            deepseek_prompts.append(user_prompt)
            deepseek_dois.append(doi)
            doi_to_data[doi] = (data, ModelTier.DEEPSEEK_V3)

    # Run batches for each tier via invoke()
    all_results: dict[str, PaperSummarySchema | None] = {}

    if deepseek_prompts:
        logger.debug(f"Submitting {len(deepseek_prompts)} papers (DEEPSEEK_V3)")
        try:
            deepseek_results = await invoke(
                tier=ModelTier.DEEPSEEK_V3,
                system=system_prompt,
                user=deepseek_prompts,
                schema=PaperSummarySchema,
                config=InvokeConfig(max_tokens=2048),
            )
            for doi, result in zip(deepseek_dois, deepseek_results, strict=True):
                all_results[doi] = result
        except Exception as e:
            logger.error(f"DeepSeek batch extraction failed: {e}")
            for doi in deepseek_dois:
                all_results[doi] = None

    if sonnet_1m_prompts:
        logger.debug(f"Submitting {len(sonnet_1m_prompts)} papers (SONNET_1M)")
        try:
            sonnet_results = await invoke(
                tier=ModelTier.SONNET_1M,
                system=system_prompt,
                user=sonnet_1m_prompts,
                schema=PaperSummarySchema,
                config=InvokeConfig(max_tokens=2048),
            )
            for doi, result in zip(sonnet_1m_dois, sonnet_results, strict=True):
                all_results[doi] = result
        except Exception as e:
            logger.error(f"Sonnet 1M batch extraction failed: {e}")
            for doi in sonnet_1m_dois:
                all_results[doi] = None

    summaries = {}
    es_ids = {}
    zotero_keys = {}

    for doi in paper_data.keys():
        data, tier = doi_to_data[doi]
        paper = data["paper"]
        extracted = all_results.get(doi)

        if extracted is not None:
            try:
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
            logger.warning(f"Summary extraction failed for {doi}: No result returned")
            failed_dois.add(doi)

    logger.info(f"Extracted {len(summaries)} summaries, {len(failed_dois)} failed")
    return summaries, es_ids, zotero_keys, failed_dois
