"""Find right-now hooks via date-filtered Perplexity search.

For recency-high publications, searches for concrete recent findings
(last 14-21 days) that deep-dive writers can anchor their openings on.
Runs in parallel (one instance per deep-dive) alongside fetch_content.
"""

import asyncio
import logging
import re
from datetime import date, timedelta
from typing import Any, Literal

from langsmith import traceable

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

from ..state import RightNowHook

logger = logging.getLogger(__name__)

SEARCH_LOOKBACK_DAYS = 21


async def _generate_queries(title: str, theme: str) -> list[str]:
    """Generate 2 targeted search queries from a deep-dive's plan."""
    response = await invoke(
        tier=ModelTier.HAIKU,
        system=(
            "Generate exactly 2 web search queries to find the most recent "
            "developments (last 2-3 weeks) related to this research topic. "
            "Queries should target recent news, preprints, reports, or data "
            "releases — not textbook overviews. Return ONLY the 2 queries, "
            "one per line, no numbering or commentary."
        ),
        user=f"Deep-dive title: {title}\nTheme: {theme}",
        config=InvokeConfig(max_tokens=200, batch_policy=BatchPolicy.PREFER_SPEED),
    )
    text = response.content if isinstance(response.content, str) else str(response.content)
    queries = [q.strip() for q in text.strip().splitlines() if q.strip()]
    return queries[:2]


async def _search_perplexity(query: str, after_date: str) -> list[dict]:
    """Run a single Perplexity search with date filtering."""
    from langchain_tools.perplexity import perplexity_search

    try:
        result = await perplexity_search.ainvoke({
            "query": query,
            "limit": 10,
            "after_date": after_date,
        })
        return result.get("results", [])
    except Exception as e:
        logger.warning(f"Perplexity search failed for '{query}': {e}")
        return []


async def _fetch_url_content(url: str) -> str:
    """Fetch full content from a URL using the unified scraping system."""
    from core.scraping.unified import get_url

    try:
        result = await get_url(url)
        return result.content
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return ""


async def _synthesize_hooks(
    deep_dive_id: str,
    title: str,
    theme: str,
    search_results: list[dict],
) -> list[dict]:
    """Use Sonnet to select and synthesize 1-3 right-now hooks from search results."""
    if not search_results:
        return []

    # Format results for the LLM
    results_text = []
    for i, r in enumerate(search_results, 1):
        entry = f"{i}. {r.get('title', 'Untitled')}"
        if r.get("url"):
            entry += f"\n   URL: {r['url']}"
        if r.get("date"):
            entry += f"\n   Date: {r['date']}"
        if r.get("snippet"):
            entry += f"\n   Snippet: {r['snippet']}"
        results_text.append(entry)

    response = await invoke(
        tier=ModelTier.SONNET,
        system=(
            "You are selecting the most relevant recent findings for a deep-dive "
            "article. From the search results below, pick 1-3 that represent "
            "concrete, specific, recent developments directly relevant to the "
            "deep-dive's theme. Skip results that are generic, off-topic, or just "
            "restate well-known facts.\n\n"
            "For each selected result, provide:\n"
            "- The result number\n"
            "- The actual publication date (check the URL path for year/month clues — "
            "e.g., /2023/12/ means December 2023, not 2026. If the search-reported "
            "date conflicts with the URL path date, use the URL path date. "
            "If uncertain, write 'date uncertain'.)\n"
            "- A 2-3 sentence summary of the specific finding and why it matters "
            "for this deep-dive\n\n"
            "Format each as:\n"
            "HOOK <number>\n"
            "DATE: <actual publication date>\n"
            "<summary>\n\n"
            "Skip results whose actual publication date is more than 3 months old "
            "(even if the search reported them as recent).\n"
            "If none of the results are genuinely relevant and recent, respond "
            "with NONE."
        ),
        user=(
            f"## Deep-Dive Plan\n"
            f"Title: {title}\n"
            f"Theme: {theme}\n\n"
            f"## Search Results\n\n" + "\n\n".join(results_text)
        ),
        config=InvokeConfig(max_tokens=1000, batch_policy=BatchPolicy.PREFER_BALANCE),
    )

    text = response.content if isinstance(response.content, str) else str(response.content)

    if "NONE" in text and len(text.strip()) < 20:
        return []

    # Parse HOOK blocks (with optional DATE line)
    hooks = []
    for match in re.finditer(r"HOOK\s+(\d+)\s*\n(.*?)(?=HOOK\s+\d+|$)", text, re.DOTALL):
        idx = int(match.group(1)) - 1
        block = match.group(2).strip()
        if 0 <= idx < len(search_results) and block:
            r = search_results[idx]
            # Extract DATE line if present
            date_match = re.match(r"DATE:\s*(.+)\n(.*)", block, re.DOTALL)
            if date_match:
                verified_date = date_match.group(1).strip()
                summary = date_match.group(2).strip()
            else:
                verified_date = r.get("date", "")
                summary = block
            if summary:
                hooks.append({
                    "finding": summary,
                    "source_title": r.get("title", ""),
                    "source_url": r.get("url", ""),
                    "source_date": verified_date,
                })

    return hooks[:3]


@traceable(run_type="chain", name="EveningReads_FindRightNow")
async def find_right_now_node(state: dict) -> dict[str, Any]:
    """Find right-now hooks for a single deep-dive via Perplexity search.

    Called via Send() with assignment-specific data. Runs date-filtered
    web searches and synthesizes concrete recent findings.

    Expected state keys from Send():
        - deep_dive_id: Which deep-dive this is
        - title: Article title
        - theme: Article theme description

    Returns:
        State update with right_now_hooks list (aggregated via add reducer)
    """
    deep_dive_id: Literal["deep_dive_1", "deep_dive_2", "deep_dive_3"] = state.get("deep_dive_id")
    title = state.get("title", "")
    theme = state.get("theme", "")

    if not deep_dive_id:
        return {"errors": [{"node": "find_right_now", "error": "Missing deep_dive_id"}]}

    logger.info(f"Finding right-now hooks for {deep_dive_id}: '{title}'")

    # 1. Generate search queries
    queries = await _generate_queries(title, theme)
    if not queries:
        logger.warning(f"No queries generated for {deep_dive_id}")
        return {"right_now_hooks": []}

    # 2. Run Perplexity searches in parallel
    after_date = (date.today() - timedelta(days=SEARCH_LOOKBACK_DAYS)).isoformat()
    search_tasks = [_search_perplexity(q, after_date) for q in queries]
    results_lists = await asyncio.gather(*search_tasks)

    # Flatten and deduplicate by URL
    seen_urls: set[str] = set()
    all_results = []
    for results in results_lists:
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    logger.info(f"Found {len(all_results)} unique results for {deep_dive_id}")

    if not all_results:
        return {"right_now_hooks": []}

    # 3. Synthesize hooks with Sonnet
    raw_hooks = await _synthesize_hooks(deep_dive_id, title, theme, all_results)

    if not raw_hooks:
        logger.info(f"No relevant hooks found for {deep_dive_id}")
        return {"right_now_hooks": []}

    # 4. Fetch URL content for selected hooks in parallel
    fetch_tasks = [_fetch_url_content(h["source_url"]) for h in raw_hooks]
    contents = await asyncio.gather(*fetch_tasks)

    hooks: list[RightNowHook] = []
    for hook_data, content in zip(raw_hooks, contents):
        hooks.append(
            RightNowHook(
                deep_dive_id=deep_dive_id,
                finding=hook_data["finding"],
                source_title=hook_data["source_title"],
                source_url=hook_data["source_url"],
                source_date=hook_data["source_date"],
                content=content,
            )
        )

    logger.info(f"Produced {len(hooks)} right-now hooks for {deep_dive_id}")
    return {"right_now_hooks": hooks}
