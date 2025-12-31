"""Relevance scoring utilities for academic literature review workflow.

Contains:
- Relevance scoring prompts and functions
- Batch scoring with concurrency control
"""

import logging

from workflows.research.subgraphs.academic_lit_review.state import PaperMetadata
from workflows.shared.llm_utils import (
    ModelTier,
    get_llm,
    invoke_with_cache,
)

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
    tier: ModelTier = ModelTier.HAIKU,
) -> tuple[float, str]:
    """Score a single paper's relevance to the research topic.

    Args:
        paper: Paper metadata to evaluate
        topic: Research topic
        research_questions: List of research questions
        tier: Model tier for scoring

    Returns:
        Tuple of (relevance_score, reasoning)
    """
    import json

    llm = get_llm(tier=tier)

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
            system_prompt=RELEVANCE_SCORING_SYSTEM,
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
    tier: ModelTier = ModelTier.HAIKU,
    max_concurrent: int = 10,
) -> tuple[list[PaperMetadata], list[PaperMetadata]]:
    """Score multiple papers' relevance and filter by threshold.

    Args:
        papers: Papers to evaluate
        topic: Research topic
        research_questions: List of research questions
        threshold: Minimum relevance score to include
        tier: Model tier for scoring
        max_concurrent: Maximum concurrent scoring calls

    Returns:
        Tuple of (relevant_papers, rejected_papers)
    """
    import asyncio

    if not papers:
        return [], []

    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_with_limit(paper: PaperMetadata) -> tuple[PaperMetadata, float, str]:
        async with semaphore:
            score, reasoning = await score_paper_relevance(
                paper, topic, research_questions, tier
            )
            return paper, score, reasoning

    # Score all papers concurrently
    tasks = [score_with_limit(paper) for paper in papers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    relevant = []
    rejected = []

    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Relevance scoring task failed: {result}")
            continue

        paper, score, reasoning = result
        # Attach score to paper for downstream filtering
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
