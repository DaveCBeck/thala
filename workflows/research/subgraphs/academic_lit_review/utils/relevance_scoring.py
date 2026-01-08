"""Relevance scoring utilities for academic literature review workflow.

Contains:
- Relevance scoring prompts and functions
- Batch scoring using Anthropic Batch API (50% cost reduction)
"""

import json
import logging

from workflows.research.subgraphs.academic_lit_review.state import PaperMetadata
from workflows.shared.llm_utils import (
    ModelTier,
    get_llm,
    invoke_with_cache,
)
from workflows.shared.batch_processor import BatchProcessor
from workflows.shared.language import LanguageConfig, get_translated_prompt

logger = logging.getLogger(__name__)


RELEVANCE_SCORING_SYSTEM = """You are an academic literature review assistant evaluating paper relevance.

Given a research topic and a paper's metadata, score its relevance from 0.0 to 1.0:
- 1.0: Directly addresses the core topic, essential reading
- 0.8-0.9: Highly relevant, addresses key aspects
- 0.6-0.7: Moderately relevant, provides useful context
- 0.4-0.5: Tangentially related, may have some value
- 0.2-0.3: Loosely related, minimal direct relevance
- 0.0-0.1: Not relevant to the topic

Consider:
- Title and abstract alignment with topic
- Methodology relevance (if applicable)
- Theoretical framework fit
- Disciplinary alignment

Output ONLY a JSON object with:
{
  "relevance_score": <float 0.0-1.0>,
  "reasoning": "<brief 1-2 sentence explanation>"
}"""

RELEVANCE_SCORING_USER_TEMPLATE = """Research Topic: {topic}
Research Questions: {research_questions}

Paper to Evaluate:
- Title: {title}
- Authors: {authors}
- Year: {year}
- Venue: {venue}
- Abstract: {abstract}
- Primary Topic: {primary_topic}

Evaluate this paper's relevance to the research topic."""


async def score_paper_relevance(
    paper: PaperMetadata,
    topic: str,
    research_questions: list[str],
    language_config: LanguageConfig | None = None,
    tier: ModelTier = ModelTier.HAIKU,
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
    import json

    llm = get_llm(tier=tier)

    # Translate system prompt if needed
    system_prompt = RELEVANCE_SCORING_SYSTEM
    if language_config and language_config["code"] != "en":
        system_prompt = await get_translated_prompt(
            RELEVANCE_SCORING_SYSTEM,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_relevance_system",
        )

    # Format authors
    authors_str = ", ".join(
        a.get("name", "") for a in paper.get("authors", [])[:5]
    )
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

        content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        content = content.strip()

        # Parse JSON response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        result = json.loads(content)
        score = float(result.get("relevance_score", 0.5))
        reasoning = result.get("reasoning", "")

        # Clamp score to valid range
        score = max(0.0, min(1.0, score))

        return score, reasoning

    except Exception as e:
        logger.warning(f"Failed to score relevance for {paper.get('title', 'Unknown')}: {e}")
        # Default to moderate relevance on failure
        return 0.5, f"Scoring failed: {e}"


async def batch_score_relevance(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    threshold: float = 0.6,
    language_config: LanguageConfig | None = None,
    tier: ModelTier = ModelTier.HAIKU,
    max_concurrent: int = 10,  # Kept for API compatibility, not used with batching
    use_batch_api: bool = True,  # Set False for rapid iteration (skips batch API)
) -> tuple[list[PaperMetadata], list[PaperMetadata]]:
    """Score multiple papers' relevance and filter by threshold.

    Uses Anthropic Batch API for 50% cost reduction when scoring 5+ papers.
    Falls back to concurrent individual calls for smaller batches.

    Args:
        papers: Papers to evaluate
        topic: Research topic
        research_questions: List of research questions
        threshold: Minimum relevance score to include
        language_config: Optional language configuration for translation
        tier: Model tier for scoring
        max_concurrent: Kept for API compatibility

    Returns:
        Tuple of (relevant_papers, rejected_papers)
    """
    import asyncio

    if not papers:
        return [], []

    # Use batch API for 5+ papers (50% cost reduction) when enabled
    if use_batch_api and len(papers) >= 5:
        return await _batch_score_relevance_batched(
            papers, topic, research_questions, threshold, language_config, tier
        )

    # Fall back to concurrent calls for small batches
    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_with_limit(paper: PaperMetadata) -> tuple[PaperMetadata, float, str]:
        async with semaphore:
            score, reasoning = await score_paper_relevance(
                paper, topic, research_questions, language_config, tier
            )
            return paper, score, reasoning

    tasks = [score_with_limit(paper) for paper in papers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    relevant = []
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
                f"RELEVANT ({score:.2f}): {paper.get('title', 'Unknown')[:50]} - {reasoning}"
            )
        else:
            rejected.append(paper)
            logger.debug(
                f"REJECTED ({score:.2f}): {paper.get('title', 'Unknown')[:50]} - {reasoning}"
            )

    logger.info(
        f"Relevance filtering: {len(relevant)} relevant, {len(rejected)} rejected "
        f"(threshold={threshold})"
    )

    return relevant, rejected


async def _batch_score_relevance_batched(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    threshold: float,
    language_config: LanguageConfig | None,
    tier: ModelTier,
) -> tuple[list[PaperMetadata], list[PaperMetadata]]:
    """Score papers using Anthropic Batch API for 50% cost reduction."""
    # Translate system prompt if needed
    system_prompt = RELEVANCE_SCORING_SYSTEM
    if language_config and language_config["code"] != "en":
        system_prompt = await get_translated_prompt(
            RELEVANCE_SCORING_SYSTEM,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_relevance_system",
        )

    # Build batch requests
    processor = BatchProcessor(poll_interval=30)
    paper_index = {}  # Map custom_id back to paper

    for i, paper in enumerate(papers):
        custom_id = f"relevance-{i}"
        paper_index[custom_id] = paper

        # Format authors
        authors_str = ", ".join(
            a.get("name", "") for a in paper.get("authors", [])[:5]
        )
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

        processor.add_request(
            custom_id=custom_id,
            prompt=user_prompt,
            model=tier,
            max_tokens=512,
            system=system_prompt,
        )

    logger.info(f"Submitting batch of {len(papers)} papers for relevance scoring")
    results = await processor.execute_batch()

    relevant = []
    rejected = []

    for custom_id, paper in paper_index.items():
        result = results.get(custom_id)
        if not result or not result.success:
            error_msg = result.error if result else "No result returned"
            logger.warning(f"Relevance scoring failed for {paper.get('title', 'Unknown')[:50]}: {error_msg}")
            paper["relevance_score"] = 0.5
            rejected.append(paper)
            continue

        try:
            content = result.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            parsed = json.loads(content)
            score = float(parsed.get("relevance_score", 0.5))
            reasoning = parsed.get("reasoning", "")
            score = max(0.0, min(1.0, score))

            paper["relevance_score"] = score
            if score >= threshold:
                relevant.append(paper)
                logger.debug(
                    f"RELEVANT ({score:.2f}): {paper.get('title', 'Unknown')[:50]} - {reasoning}"
                )
            else:
                rejected.append(paper)
                logger.debug(
                    f"REJECTED ({score:.2f}): {paper.get('title', 'Unknown')[:50]} - {reasoning}"
                )

        except Exception as e:
            logger.warning(f"Failed to parse relevance result for {paper.get('title', 'Unknown')[:50]}: {e}")
            paper["relevance_score"] = 0.5
            rejected.append(paper)

    logger.info(
        f"Relevance filtering (batch): {len(relevant)} relevant, {len(rejected)} rejected "
        f"(threshold={threshold})"
    )

    return relevant, rejected
