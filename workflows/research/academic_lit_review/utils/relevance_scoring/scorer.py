"""Main relevance scoring logic for academic papers.

Routes through unified invoke() layer for consistent LLM access.
"""

import asyncio
import logging

from core.llm_broker import BatchPolicy
from workflows.research.academic_lit_review.state import PaperMetadata
from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier
from workflows.shared.llm_utils.models import is_deepseek_tier
from workflows.shared.language import LanguageConfig, get_translated_prompt

from .strategies import chunk_papers, format_paper_for_batch
from .types import (
    BATCH_RELEVANCE_SCORING_SYSTEM,
    BATCH_RELEVANCE_SCORING_USER_TEMPLATE,
    BatchRelevanceScores,
    RELEVANCE_SCORING_SYSTEM,
    RELEVANCE_SCORING_USER_TEMPLATE,
    RelevanceScore,
)

logger = logging.getLogger(__name__)


async def score_paper_relevance(
    paper: PaperMetadata,
    topic: str,
    research_questions: list[str],
    language_config: LanguageConfig | None = None,
    tier: ModelTier = ModelTier.DEEPSEEK_V3,
) -> tuple[float, str]:
    """Score a single paper's relevance to the research topic.

    Routes through unified invoke() layer. DeepSeek tiers are routed directly
    (fixing the previous 404 bug), Anthropic tiers can use broker batching.

    Args:
        paper: Paper metadata to evaluate
        topic: Research topic
        research_questions: List of research questions
        language_config: Optional language configuration for translation
        tier: Model tier for scoring

    Returns:
        Tuple of (relevance_score, reasoning)
    """
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
        # DeepSeek doesn't support broker batching, Anthropic models use PREFER_BALANCE
        # invoke() handles routing: DeepSeek goes direct, Anthropic can use broker
        batch_policy = None if is_deepseek_tier(tier) else BatchPolicy.PREFER_BALANCE

        result = await invoke(
            tier=tier,
            system=system_prompt,
            user=user_prompt,
            schema=RelevanceScore,
            config=InvokeConfig(
                max_tokens=512,
                batch_policy=batch_policy,
                cache=not is_deepseek_tier(tier),  # Only cache for Anthropic
            ),
        )

        # Result is already validated by Pydantic, but clamp just in case
        score = max(0.0, min(1.0, result.relevance_score))
        return score, result.reasoning

    except Exception as e:
        logger.warning(f"Failed to score relevance for {paper.get('title', 'Unknown')}: {e}")
        return 0.5, f"Scoring failed: {e}"


async def batch_score_relevance(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    threshold: float = 0.6,
    fallback_threshold: float = 0.5,
    language_config: LanguageConfig | None = None,
    tier: ModelTier = ModelTier.DEEPSEEK_V3,
    chunk_size: int = 10,
) -> tuple[list[PaperMetadata], list[PaperMetadata], list[PaperMetadata]]:
    """Score multiple papers' relevance and filter by threshold.

    Routes through unified invoke() layer. DeepSeek tiers are routed directly
    (fixing the previous 404 bug), Anthropic tiers can use broker batching.
    Uses chunked prompts for better calibration through cross-comparison.

    Args:
        papers: Papers to evaluate
        topic: Research topic
        research_questions: List of research questions
        threshold: Minimum relevance score to include (default 0.6)
        fallback_threshold: Minimum score for fallback eligibility (default 0.5)
        language_config: Optional language configuration for translation
        tier: Model tier for scoring
        chunk_size: Papers per chunk for cross-comparison (default 10)

    Returns:
        Tuple of (relevant_papers, fallback_candidates, rejected_papers)
        - relevant: score >= threshold
        - fallback_candidates: fallback_threshold <= score < threshold
        - rejected: score < fallback_threshold
    """
    if not papers:
        return [], [], []

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
    research_questions_str = "; ".join(research_questions[:3])

    # DeepSeek doesn't support broker batching
    batch_policy = None if is_deepseek_tier(tier) else BatchPolicy.PREFER_BALANCE
    use_cache = not is_deepseek_tier(tier)

    logger.info(
        f"Scoring {len(papers)} papers via invoke() ({len(chunks)} chunks of up to {chunk_size}, topic: {topic[:50]}...)"
    )

    async def score_chunk(chunk_idx: int, chunk: list[PaperMetadata]) -> tuple[int, list[PaperMetadata], BatchRelevanceScores | None, str | None]:
        """Score a single chunk, returning (index, papers, result, error)."""
        papers_text = "\n".join(format_paper_for_batch(p) for p in chunk)

        user_prompt = BATCH_RELEVANCE_SCORING_USER_TEMPLATE.format(
            topic=topic,
            research_questions=research_questions_str,
            papers=papers_text,
        )

        try:
            result = await invoke(
                tier=tier,
                system=system_prompt,
                user=user_prompt,
                schema=BatchRelevanceScores,
                config=InvokeConfig(
                    max_tokens=min(4096, 150 * len(chunk) + 100),
                    batch_policy=batch_policy,
                    cache=use_cache,
                ),
            )
            return chunk_idx, chunk, result, None
        except Exception as e:
            return chunk_idx, chunk, None, str(e)

    # Process all chunks concurrently
    chunk_results = await asyncio.gather(
        *[score_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    )

    # Parse results
    relevant = []
    fallback_candidates = []
    rejected = []

    for chunk_idx, papers_in_chunk, result, error in chunk_results:
        if error or result is None:
            logger.warning(f"Failed to score chunk-{chunk_idx}: {error}, defaulting papers to 0.5")
            for paper in papers_in_chunk:
                paper["relevance_score"] = 0.5
                fallback_candidates.append(paper)
            continue

        # Build DOI -> (score, reasoning) map from structured result
        doi_scores: dict[str, tuple[float, str]] = {}
        for item in result.papers:
            score = max(0.0, min(1.0, item.relevance_score))
            doi_scores[item.doi] = (score, item.reasoning)

        # Match papers to their scores
        for paper in papers_in_chunk:
            paper_doi = paper.get("doi", "")
            if paper_doi in doi_scores:
                score, reasoning = doi_scores[paper_doi]
            else:
                logger.warning(f"DOI {paper_doi} not found in chunk response, defaulting to 0.5")
                score = 0.5

            paper["relevance_score"] = score
            if score >= threshold:
                relevant.append(paper)
                logger.debug(f"Relevant ({score:.2f}): {paper.get('title', 'Unknown')[:50]}")
            elif score >= fallback_threshold:
                fallback_candidates.append(paper)
                logger.debug(f"Fallback ({score:.2f}): {paper.get('title', 'Unknown')[:50]}")
            else:
                rejected.append(paper)
                logger.debug(f"Rejected ({score:.2f}): {paper.get('title', 'Unknown')[:50]}")

    logger.info(
        f"Relevance filtering: {len(relevant)} relevant, "
        f"{len(fallback_candidates)} fallback, {len(rejected)} rejected "
        f"(threshold={threshold}, fallback_threshold={fallback_threshold})"
    )

    return relevant, fallback_candidates, rejected
