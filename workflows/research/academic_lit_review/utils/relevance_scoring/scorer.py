"""Main relevance scoring logic for academic papers."""

import json
import logging
import os

from workflows.research.academic_lit_review.state import PaperMetadata
from workflows.shared.llm_utils import (
    ModelTier,
    get_llm,
    invoke_with_cache,
    batch_invoke_with_cache,
)
from workflows.shared.llm_utils.models import is_deepseek_tier
from workflows.shared.batch_processor import BatchProcessor
from workflows.shared.language import LanguageConfig, get_translated_prompt
from .types import (
    RELEVANCE_SCORING_SYSTEM,
    BATCH_RELEVANCE_SCORING_SYSTEM,
    RELEVANCE_SCORING_USER_TEMPLATE,
    BATCH_RELEVANCE_SCORING_USER_TEMPLATE,
)
from .strategies import format_paper_for_batch, chunk_papers

logger = logging.getLogger(__name__)


def _get_batch_api_default() -> bool:
    """Get default for batch API from environment variable."""
    return os.getenv("THALA_PREFER_BATCH_API", "").lower() in ("true", "1", "yes")


async def score_paper_relevance(
    paper: PaperMetadata,
    topic: str,
    research_questions: list[str],
    language_config: LanguageConfig | None = None,
    tier: ModelTier = ModelTier.DEEPSEEK_V3,
) -> tuple[float, str]:
    """Score a single paper's relevance to the research topic.

    Args:
        paper: Paper metadata to evaluate
        topic: Research topic
        research_questions: List of research questions
        language_config: Optional language configuration for translation
        tier: Model tier for scoring

    Returns:
        Tuple of (relevance_score, reasoning)
    """
    llm = get_llm(tier=tier)

    system_prompt = RELEVANCE_SCORING_SYSTEM
    if language_config and language_config["code"] != "en":
        system_prompt = await get_translated_prompt(
            RELEVANCE_SCORING_SYSTEM,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_relevance_system",
        )

    authors_str = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:5])
    if len(paper.get("authors", [])) > 5:
        authors_str += " et al."

    user_prompt = RELEVANCE_SCORING_USER_TEMPLATE.format(
        topic=topic,
        research_questions="; ".join(research_questions[:3]),
        title=paper.get("title", "Unknown"),
        authors=authors_str or "Unknown",
        year=paper.get("year", "Unknown"),
        venue=paper.get("venue", "Unknown"),
        abstract=(paper.get("abstract") or "No abstract available")[:1000],
        primary_topic=paper.get("primary_topic", "Not specified"),
    )

    try:
        response = await invoke_with_cache(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        content = (
            response.content
            if isinstance(response.content, str)
            else response.content[0].get("text", "")
        )
        content = content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        result = json.loads(content)
        score = float(result.get("relevance_score", 0.5))
        reasoning = result.get("reasoning", "")

        score = max(0.0, min(1.0, score))

        return score, reasoning

    except Exception as e:
        logger.warning(
            f"Failed to score relevance for {paper.get('title', 'Unknown')}: {e}"
        )
        return 0.5, f"Scoring failed: {e}"


async def batch_score_relevance(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    threshold: float = 0.6,
    fallback_threshold: float = 0.5,
    language_config: LanguageConfig | None = None,
    tier: ModelTier = ModelTier.DEEPSEEK_V3,
    max_concurrent: int = 10,
    use_batch_api: bool | None = None,
) -> tuple[list[PaperMetadata], list[PaperMetadata], list[PaperMetadata]]:
    """Score multiple papers' relevance and filter by threshold.

    Uses Anthropic Batch API for 50% cost reduction when scoring 5+ papers.
    Falls back to concurrent individual calls for smaller batches.

    Args:
        papers: Papers to evaluate
        topic: Research topic
        research_questions: List of research questions
        threshold: Minimum relevance score to include (default 0.6)
        fallback_threshold: Minimum score for fallback eligibility (default 0.5)
        language_config: Optional language configuration for translation
        tier: Model tier for scoring
        max_concurrent: Kept for API compatibility
        use_batch_api: Whether to use batch API; defaults to THALA_PREFER_BATCH_API env var

    Returns:
        Tuple of (relevant_papers, fallback_candidates, rejected_papers)
        - relevant: score >= threshold
        - fallback_candidates: fallback_threshold <= score < threshold
        - rejected: score < fallback_threshold
    """
    import asyncio

    if not papers:
        return [], [], []

    # Resolve batch API preference from env var if not explicitly set
    if use_batch_api is None:
        use_batch_api = _get_batch_api_default()

    # Disable batch API for DeepSeek models (no batch API available)
    effective_use_batch = use_batch_api and not is_deepseek_tier(tier)

    if effective_use_batch and len(papers) >= 5:
        return await _batch_score_relevance_batched(
            papers, topic, research_questions, threshold, fallback_threshold, language_config, tier
        )

    # DeepSeek: use cache-aware batch processing (first request -> wait -> batch)
    if is_deepseek_tier(tier):
        return await _batch_score_deepseek_cached(
            papers, topic, research_questions, threshold, fallback_threshold, language_config, tier, max_concurrent
        )

    # Non-DeepSeek fallback: concurrent calls
    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_with_limit(
        paper: PaperMetadata,
    ) -> tuple[PaperMetadata, float, str]:
        async with semaphore:
            score, reasoning = await score_paper_relevance(
                paper, topic, research_questions, language_config, tier
            )
            return paper, score, reasoning

    tasks = [score_with_limit(paper) for paper in papers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    relevant = []
    fallback_candidates = []
    rejected = []

    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Relevance scoring task failed: {result}")
            continue

        paper, score, reasoning = result
        paper["relevance_score"] = score
        if score >= threshold:
            relevant.append(paper)
            logger.debug(
                f"Relevant ({score:.2f}): {paper.get('title', 'Unknown')[:50]}"
            )
        elif score >= fallback_threshold:
            fallback_candidates.append(paper)
            logger.debug(
                f"Fallback ({score:.2f}): {paper.get('title', 'Unknown')[:50]}"
            )
        else:
            rejected.append(paper)
            logger.debug(
                f"Rejected ({score:.2f}): {paper.get('title', 'Unknown')[:50]}"
            )

    logger.info(
        f"Relevance filtering: {len(relevant)} relevant, {len(fallback_candidates)} fallback, "
        f"{len(rejected)} rejected (threshold={threshold}, fallback_threshold={fallback_threshold})"
    )

    return relevant, fallback_candidates, rejected


async def _batch_score_deepseek_cached(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    threshold: float,
    fallback_threshold: float,
    language_config: LanguageConfig | None,
    tier: ModelTier,
    max_concurrent: int,
    chunk_size: int = 10,
) -> tuple[list[PaperMetadata], list[PaperMetadata], list[PaperMetadata]]:
    """Score papers using DeepSeek with cache-aware chunked batch processing.

    Chunks papers into groups of chunk_size for cross-comparison scoring,
    then uses cache warmup coordination:
    1. Send first chunk to warm the cache
    2. Wait for cache construction (~10s)
    3. Process remaining chunks concurrently (cache hits)

    This combines better calibration (cross-comparison within chunks) with
    ~90% cost reduction on input tokens for chunks 2-N.

    Returns:
        Tuple of (relevant_papers, fallback_candidates, rejected_papers)
    """
    llm = get_llm(tier=tier)

    system_prompt = BATCH_RELEVANCE_SCORING_SYSTEM
    if language_config and language_config["code"] != "en":
        system_prompt = await get_translated_prompt(
            BATCH_RELEVANCE_SCORING_SYSTEM,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_batch_relevance_system",
        )

    # Chunk papers for cross-comparison scoring
    chunks = chunk_papers(papers, chunk_size)
    chunk_index: dict[str, list[PaperMetadata]] = {}

    # Build cache prefix (shared portion of user prompts)
    research_questions_str = "; ".join(research_questions[:3])
    cache_prefix = f"Research Topic: {topic}\nResearch Questions: {research_questions_str}\n\nPapers to Evaluate:"

    # Build user prompts for each chunk
    user_prompts: list[tuple[str, str]] = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk-{i}"
        chunk_index[chunk_id] = chunk

        papers_text = "\n".join(format_paper_for_batch(p) for p in chunk)

        user_prompt = BATCH_RELEVANCE_SCORING_USER_TEMPLATE.format(
            topic=topic,
            research_questions=research_questions_str,
            papers=papers_text,
        )
        user_prompts.append((chunk_id, user_prompt))

    logger.info(
        f"Scoring {len(papers)} papers with DeepSeek cache-aware batch "
        f"({len(chunks)} chunks of up to {chunk_size}, topic: {topic[:50]}...)"
    )

    # Process with cache warmup coordination
    responses = await batch_invoke_with_cache(
        llm,
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        cache_prefix=cache_prefix,
        max_concurrent=max_concurrent,
    )

    # Parse results
    relevant = []
    fallback_candidates = []
    rejected = []

    for chunk_id, papers_in_chunk in chunk_index.items():
        response = responses.get(chunk_id)
        if not response:
            logger.warning(f"No response for {chunk_id}, defaulting papers to 0.5")
            for paper in papers_in_chunk:
                paper["relevance_score"] = 0.5
                fallback_candidates.append(paper)  # 0.5 is at fallback threshold
            continue

        try:
            content = (
                response.content
                if isinstance(response.content, str)
                else response.content[0].get("text", "")
            )
            content = content.strip()

            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            parsed = json.loads(content)

            # Build DOI -> (score, reasoning) map from response
            doi_scores: dict[str, tuple[float, str]] = {}
            for item in parsed:
                doi = item.get("doi", "")
                score = float(item.get("relevance_score", 0.5))
                reasoning = item.get("reasoning", "")
                score = max(0.0, min(1.0, score))
                doi_scores[doi] = (score, reasoning)

            # Match papers to their scores
            for paper in papers_in_chunk:
                paper_doi = paper.get("doi", "")
                if paper_doi in doi_scores:
                    score, reasoning = doi_scores[paper_doi]
                else:
                    logger.warning(
                        f"DOI {paper_doi} not found in chunk response, defaulting to 0.5"
                    )
                    score, reasoning = 0.5, "DOI not in response"

                paper["relevance_score"] = score
                if score >= threshold:
                    relevant.append(paper)
                    logger.debug(
                        f"Relevant ({score:.2f}): {paper.get('title', 'Unknown')[:50]}"
                    )
                elif score >= fallback_threshold:
                    fallback_candidates.append(paper)
                    logger.debug(
                        f"Fallback ({score:.2f}): {paper.get('title', 'Unknown')[:50]}"
                    )
                else:
                    rejected.append(paper)
                    logger.debug(
                        f"Rejected ({score:.2f}): {paper.get('title', 'Unknown')[:50]}"
                    )

        except Exception as e:
            logger.warning(
                f"Failed to parse relevance for {chunk_id}: {e}"
            )
            for paper in papers_in_chunk:
                paper["relevance_score"] = 0.5
                fallback_candidates.append(paper)  # 0.5 is at fallback threshold

    logger.info(
        f"Relevance filtering (DeepSeek cached): {len(relevant)} relevant, "
        f"{len(fallback_candidates)} fallback, {len(rejected)} rejected "
        f"(threshold={threshold}, fallback_threshold={fallback_threshold})"
    )

    return relevant, fallback_candidates, rejected


async def _batch_score_relevance_batched(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    threshold: float,
    fallback_threshold: float,
    language_config: LanguageConfig | None,
    tier: ModelTier,
    chunk_size: int = 10,
) -> tuple[list[PaperMetadata], list[PaperMetadata], list[PaperMetadata]]:
    """Score papers using Anthropic Batch API with chunked prompts.

    Groups papers into chunks of chunk_size for more efficient scoring.
    Each batch request scores multiple papers, reducing API overhead
    and enabling better calibration through comparison.

    Returns:
        Tuple of (relevant_papers, fallback_candidates, rejected_papers)
    """
    system_prompt = BATCH_RELEVANCE_SCORING_SYSTEM
    if language_config and language_config["code"] != "en":
        system_prompt = await get_translated_prompt(
            BATCH_RELEVANCE_SCORING_SYSTEM,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_batch_relevance_system",
        )

    processor = BatchProcessor(poll_interval=30)
    chunks = chunk_papers(papers, chunk_size)
    chunk_index: dict[str, list[PaperMetadata]] = {}

    for i, chunk in enumerate(chunks):
        custom_id = f"relevance-chunk-{i}"
        chunk_index[custom_id] = chunk

        papers_text = "\n".join(format_paper_for_batch(p) for p in chunk)

        user_prompt = BATCH_RELEVANCE_SCORING_USER_TEMPLATE.format(
            topic=topic,
            research_questions="; ".join(research_questions[:3]),
            papers=papers_text,
        )

        max_tokens = min(4096, 150 * len(chunk) + 100)

        processor.add_request(
            custom_id=custom_id,
            prompt=user_prompt,
            model=tier,
            max_tokens=max_tokens,
            system=system_prompt,
        )

    logger.info(
        f"Submitting batch of {len(papers)} papers for relevance scoring "
        f"({len(chunks)} chunks of up to {chunk_size})"
    )
    results = await processor.execute_batch()

    relevant = []
    fallback_candidates = []
    rejected = []

    for custom_id, papers_in_chunk in chunk_index.items():
        result = results.get(custom_id)
        if not result or not result.success:
            error_msg = result.error if result else "No result returned"
            logger.warning(
                f"Relevance scoring failed for chunk {custom_id}: {error_msg}"
            )
            for paper in papers_in_chunk:
                paper["relevance_score"] = 0.5
                fallback_candidates.append(paper)  # 0.5 is at fallback threshold
            continue

        try:
            content = result.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            parsed = json.loads(content)

            doi_scores: dict[str, tuple[float, str]] = {}
            for item in parsed:
                doi = item.get("doi", "")
                score = float(item.get("relevance_score", 0.5))
                reasoning = item.get("reasoning", "")
                score = max(0.0, min(1.0, score))
                doi_scores[doi] = (score, reasoning)

            for paper in papers_in_chunk:
                paper_doi = paper.get("doi", "")
                if paper_doi in doi_scores:
                    score, reasoning = doi_scores[paper_doi]
                else:
                    logger.warning(
                        f"DOI {paper_doi} not found in batch response, defaulting to 0.5"
                    )
                    score, reasoning = 0.5, "DOI not in response"

                paper["relevance_score"] = score
                if score >= threshold:
                    relevant.append(paper)
                    logger.debug(
                        f"Relevant ({score:.2f}): {paper.get('title', 'Unknown')[:50]}"
                    )
                elif score >= fallback_threshold:
                    fallback_candidates.append(paper)
                    logger.debug(
                        f"Fallback ({score:.2f}): {paper.get('title', 'Unknown')[:50]}"
                    )
                else:
                    rejected.append(paper)
                    logger.debug(
                        f"Rejected ({score:.2f}): {paper.get('title', 'Unknown')[:50]}"
                    )

        except Exception as e:
            logger.warning(
                f"Failed to parse relevance result for chunk {custom_id}: {e}"
            )
            for paper in papers_in_chunk:
                paper["relevance_score"] = 0.5
                fallback_candidates.append(paper)  # 0.5 is at fallback threshold

    logger.info(
        f"Relevance filtering (batch): {len(relevant)} relevant, "
        f"{len(fallback_candidates)} fallback, {len(rejected)} rejected "
        f"(threshold={threshold}, fallback_threshold={fallback_threshold})"
    )

    return relevant, fallback_candidates, rejected
