"""Main relevance scoring logic for academic papers.

Routes through central LLM broker for unified cost/speed management.
"""

import json
import logging

from core.llm_broker import BatchPolicy, get_broker
from workflows.research.academic_lit_review.state import PaperMetadata
from workflows.shared.llm_utils import ModelTier
from workflows.shared.llm_utils.models import is_deepseek_tier
from workflows.shared.language import LanguageConfig, get_translated_prompt

from .strategies import chunk_papers, format_paper_for_batch
from .types import (
    BATCH_RELEVANCE_SCORING_SYSTEM,
    BATCH_RELEVANCE_SCORING_USER_TEMPLATE,
    RELEVANCE_SCORING_SYSTEM,
    RELEVANCE_SCORING_USER_TEMPLATE,
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

    Routes through central LLM broker for unified cost/speed management.

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
        broker = get_broker()
        # DeepSeek doesn't support batching, so use REQUIRE_SYNC
        # Anthropic models use PREFER_BALANCE
        policy = BatchPolicy.REQUIRE_SYNC if is_deepseek_tier(tier) else BatchPolicy.PREFER_BALANCE

        future = await broker.request(
            prompt=user_prompt,
            model=tier,
            policy=policy,
            max_tokens=512,
            system=system_prompt,
        )
        response = await future

        if not response.success:
            logger.warning(f"Failed to score relevance for {paper.get('title', 'Unknown')}: {response.error}")
            return 0.5, f"Scoring failed: {response.error}"

        content = response.content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        result = json.loads(content)
        score = float(result.get("relevance_score", 0.5))
        reasoning = result.get("reasoning", "")

        score = max(0.0, min(1.0, score))

        return score, reasoning

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

    Routes through central LLM broker for unified cost/speed management.
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
    chunk_index: dict[str, list[PaperMetadata]] = {}

    # Build user prompts for each chunk
    research_questions_str = "; ".join(research_questions[:3])

    broker = get_broker()
    pending_futures: dict[str, any] = {}

    # DeepSeek doesn't support batching, so use REQUIRE_SYNC
    # Anthropic models use PREFER_BALANCE
    policy = BatchPolicy.REQUIRE_SYNC if is_deepseek_tier(tier) else BatchPolicy.PREFER_BALANCE

    logger.info(
        f"Scoring {len(papers)} papers via broker ({len(chunks)} chunks of up to {chunk_size}, topic: {topic[:50]}...)"
    )

    async with broker.batch_group():
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk-{i}"
            chunk_index[chunk_id] = chunk

            papers_text = "\n".join(format_paper_for_batch(p) for p in chunk)

            user_prompt = BATCH_RELEVANCE_SCORING_USER_TEMPLATE.format(
                topic=topic,
                research_questions=research_questions_str,
                papers=papers_text,
            )

            future = await broker.request(
                prompt=user_prompt,
                model=tier,
                policy=policy,
                max_tokens=min(4096, 150 * len(chunk) + 100),
                system=system_prompt,
            )
            pending_futures[chunk_id] = future

    # Parse results
    relevant = []
    fallback_candidates = []
    rejected = []

    for chunk_id, papers_in_chunk in chunk_index.items():
        try:
            response = await pending_futures[chunk_id]

            if not response.success:
                logger.warning(f"No response for {chunk_id}: {response.error}, defaulting papers to 0.5")
                for paper in papers_in_chunk:
                    paper["relevance_score"] = 0.5
                    fallback_candidates.append(paper)
                continue

            content = response.content.strip()

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
                    logger.warning(f"DOI {paper_doi} not found in chunk response, defaulting to 0.5")
                    score, reasoning = 0.5, "DOI not in response"

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

        except Exception as e:
            logger.warning(f"Failed to parse relevance for {chunk_id}: {e}")
            for paper in papers_in_chunk:
                paper["relevance_score"] = 0.5
                fallback_candidates.append(paper)

    logger.info(
        f"Relevance filtering: {len(relevant)} relevant, "
        f"{len(fallback_candidates)} fallback, {len(rejected)} rejected "
        f"(threshold={threshold}, fallback_threshold={fallback_threshold})"
    )

    return relevant, fallback_candidates, rejected
