"""Web scan phase: sharpen research questions using recent web signal.

Runs Perplexity searches to discover recent developments, then refines
research questions and produces a landscape summary for the combine step.
"""

import asyncio
import json
import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, invoke

logger = logging.getLogger(__name__)

# Concurrency limit for Perplexity API calls
_SEARCH_SEMAPHORE_LIMIT = 5


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output before JSON parsing."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)
    return text


async def run_web_scan_phase(
    topic: str,
    research_questions: list[str],
    web_scan_window_days: int = 30,
) -> dict[str, Any]:
    """Run web scan to sharpen research questions with recent signal.

    Args:
        topic: Research topic
        research_questions: Original research questions
        web_scan_window_days: How far back web search looks (default 30)

    Returns:
        WebScanResult dict with:
        - augmented_research_questions: Refined questions
        - original_research_questions: Preserved originals
        - recent_landscape: Structured summary of current landscape
        - raw_results: Raw Perplexity search results
    """
    from datetime import datetime, timedelta

    from langchain_tools.perplexity import perplexity_search

    # Step 1: Generate search queries from topic + research questions
    search_queries = await _generate_search_queries(topic, research_questions)
    logger.info(f"Web scan: generated {len(search_queries)} search queries")

    # Step 2: Execute Perplexity searches concurrently
    after_date = (datetime.now() - timedelta(days=web_scan_window_days)).strftime("%Y-%m-%d")
    semaphore = asyncio.Semaphore(_SEARCH_SEMAPHORE_LIMIT)

    async def _bounded_search(query: str) -> dict:
        async with semaphore:
            return await perplexity_search.ainvoke(
                {
                    "query": query,
                    "limit": 20,
                    "after_date": after_date,
                }
            )

    raw_results = await asyncio.gather(
        *[_bounded_search(q) for q in search_queries],
        return_exceptions=True,
    )

    # Filter out exceptions
    valid_results = []
    for r in raw_results:
        if isinstance(r, Exception):
            logger.warning(f"Web scan search failed: {r}")
        else:
            valid_results.append(r)

    logger.info(f"Web scan: {len(valid_results)}/{len(search_queries)} searches succeeded")

    # Step 3: Synthesize results into augmented questions + landscape
    synthesis = await _synthesize_results(topic, research_questions, valid_results)

    return {
        "augmented_research_questions": synthesis["augmented_research_questions"],
        "original_research_questions": research_questions,
        "recent_landscape": synthesis["recent_landscape"],
        "raw_results": valid_results,
    }


async def _generate_search_queries(
    topic: str,
    research_questions: list[str],
) -> list[str]:
    """Generate targeted web search queries from topic and research questions."""
    rq_text = "\n".join(f"- {q}" for q in research_questions)

    response = await invoke(
        tier=ModelTier.SONNET,
        system=(
            "You generate targeted web search queries to discover recent developments "
            "on a research topic. Output ONLY a JSON array of query strings. "
            "Generate 2-3 queries per research question. "
            "Queries should emphasize recency: include terms like 'latest', '2026', "
            "'recent developments', 'new findings', 'update'."
        ),
        user=f"Topic: {topic}\n\nResearch questions:\n{rq_text}",
    )

    content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
    content = _strip_code_fences(content)

    try:
        queries = json.loads(content)
        if isinstance(queries, list):
            return [str(q) for q in queries]
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse search queries JSON, extracting from text")

    # Fallback: generate basic queries from research questions
    return [f"latest developments {topic} {q}" for q in research_questions[:4]]


async def _synthesize_results(
    topic: str,
    original_questions: list[str],
    search_results: list[dict],
) -> dict[str, Any]:
    """Synthesize search results into augmented questions and landscape summary."""
    # Build results text
    results_text = []
    for result in search_results:
        for item in result.get("results", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title or snippet:
                results_text.append(f"- {title}: {snippet}")

    rq_text = "\n".join(f"- {q}" for q in original_questions)
    findings_text = "\n".join(results_text[:60])  # Cap to avoid token overflow

    response = await invoke(
        tier=ModelTier.SONNET,
        system=(
            "You are a research analyst refining research questions based on recent web findings.\n\n"
            "Output ONLY valid JSON with two keys:\n"
            '- "augmented_research_questions": list of refined research questions. '
            "Preserve ALL original questions (may reword, never drop). "
            "Add 0-2 new sub-questions ONLY if major developments warrant it.\n"
            '- "recent_landscape": a brief structured summary (3-5 sentences) of '
            "the current landscape based on the web findings."
        ),
        user=(f"Topic: {topic}\n\nOriginal research questions:\n{rq_text}\n\nRecent web findings:\n{findings_text}"),
    )

    content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
    content = _strip_code_fences(content)

    try:
        data = json.loads(content)
        return {
            "augmented_research_questions": data.get("augmented_research_questions", original_questions),
            "recent_landscape": data.get("recent_landscape", ""),
        }
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse synthesis JSON, using original questions")
        return {
            "augmented_research_questions": original_questions,
            "recent_landscape": "",
        }
