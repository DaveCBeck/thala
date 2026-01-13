"""
Web researcher subgraph.

Specialized researcher for general web content using:
- Firecrawl: General web search
- Perplexity: AI-powered web search with synthesis

Uses the web-specific compression prompt that emphasizes:
- Recency and publication dates
- Domain authority (.gov, .edu, established news)
- Factual accuracy and bias detection
"""

import asyncio
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from langchain_tools.firecrawl import web_search
from langchain_tools.perplexity import perplexity_search
from workflows.research.web_research.state import (
    ResearcherState,
    ResearchFinding,
    WebSearchResult,
)
from workflows.research.web_research.prompts import (
    COMPRESS_WEB_RESEARCH_SYSTEM,
    COMPRESS_RESEARCH_USER_TEMPLATE,
    get_today_str,
)
from workflows.research.web_research.utils import load_prompts_with_translation, extract_json_from_llm_response
from workflows.research.web_research.subgraphs.researcher_base import (
    create_generate_queries,
    scrape_pages as base_scrape_pages,
)
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache

logger = logging.getLogger(__name__)

# Web researcher constants
MAX_SEARCHES = 5
MAX_SCRAPES = 3
MAX_RESULTS_PER_SOURCE = 5


async def _search_firecrawl(query: str) -> list[WebSearchResult]:
    """Search using Firecrawl."""
    results = []
    try:
        result = await web_search.ainvoke({"query": query, "limit": MAX_RESULTS_PER_SOURCE})
        for r in result.get("results", []):
            results.append(
                WebSearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    description=r.get("description"),
                    content=None,
                    source_metadata=None,
                )
            )
    except Exception as e:
        logger.warning(f"Firecrawl search failed for '{query}': {e}")
    return results


async def _search_perplexity(query: str) -> list[WebSearchResult]:
    """Search using Perplexity."""
    results = []
    try:
        result = await perplexity_search.ainvoke({"query": query, "limit": MAX_RESULTS_PER_SOURCE})
        for r in result.get("results", []):
            results.append(
                WebSearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    description=r.get("snippet"),
                    content=None,
                    source_metadata=None,
                )
            )
    except Exception as e:
        logger.warning(f"Perplexity search failed for '{query}': {e}")
    return results


async def execute_searches(state: ResearcherState) -> dict[str, Any]:
    """Execute web searches using Firecrawl and Perplexity in parallel.

    Sources:
    - Firecrawl: General web search
    - Perplexity: AI-powered web search with synthesis
    """
    queries = state.get("search_queries", [])
    all_results = []

    for query in queries[:MAX_SEARCHES]:
        # Run both web sources in parallel for each query
        search_tasks = [
            _search_firecrawl(query),
            _search_perplexity(query),
        ]

        results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect results (handling potential exceptions)
        for results in results_lists:
            if isinstance(results, list):
                all_results.extend(results)
            elif isinstance(results, Exception):
                logger.warning(f"Search source failed: {results}")

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r["url"]
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    logger.debug(
        f"Web researcher: found {len(unique_results)} unique results from {len(queries)} queries "
        f"across Firecrawl/Perplexity"
    )
    return {"search_results": unique_results[:15]}


async def scrape_pages(state: ResearcherState) -> dict[str, Any]:
    """Scrape top results for full content."""
    return await base_scrape_pages(state, max_scrapes=MAX_SCRAPES)


async def compress_findings(state: ResearcherState) -> dict[str, Any]:
    """Compress research into structured finding using web-specific prompt.

    Emphasizes:
    - Recency and publication dates
    - Domain authority
    - Bias detection
    """
    question = state["question"]
    search_results = state.get("search_results", [])
    scraped = state.get("scraped_content", [])
    language_config = state.get("language_config")
    language_code = language_config["code"] if language_config else None

    # Build raw research text
    if scraped:
        raw_research = "\n\n---\n\n".join(scraped)
    else:
        raw_research = "\n".join([
            f"- [{r['title']}]({r['url']}): {r.get('description', 'No description')}"
            for r in search_results
        ])

    system_prompt, user_template = await load_prompts_with_translation(
        COMPRESS_WEB_RESEARCH_SYSTEM,
        COMPRESS_RESEARCH_USER_TEMPLATE,
        language_config,
        "compress_web_research_system",
        "compress_research_user",
    )

    # Build dynamic user prompt
    user_prompt = user_template.format(
        date=get_today_str(),
        question=question["question"],
        raw_research=raw_research[:15000],
    )

    llm = get_llm(ModelTier.SONNET)

    try:
        # Use cached system prompt for 90% cost reduction on repeated calls
        response = await invoke_with_cache(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        finding_data = extract_json_from_llm_response(content)

        # Build URL -> original result map to preserve source_metadata
        url_to_result = {r.get("url", ""): r for r in search_results if r.get("url")}

        # Reconstruct sources, preserving source_metadata from original results
        finding_sources = []
        for s in finding_data.get("sources", []):
            url = s.get("url", "")
            original = url_to_result.get(url, {})
            finding_sources.append(
                WebSearchResult(
                    url=url,
                    title=s.get("title", original.get("title", "")),
                    description=s.get("relevance", original.get("description", "")),
                    content=original.get("content"),
                    source_metadata=original.get("source_metadata"),
                )
            )

        finding = ResearchFinding(
            question_id=question["question_id"],
            finding=finding_data.get("finding", "No finding"),
            sources=finding_sources,
            confidence=float(finding_data.get("confidence", 0.5)),
            gaps=finding_data.get("gaps", []),
            language_code=language_code,
        )

        logger.debug(
            f"Web researcher compressed finding for {question['question_id']}: "
            f"confidence={finding['confidence']:.2f}, {len(finding['gaps'])} gaps"
        )

        # Return as list for aggregation via add reducer in parent state
        return {"finding": finding, "research_findings": [finding]}

    except Exception as e:
        logger.error(f"Failed to compress web findings: {e}")

        # Fallback finding - preserve all original data
        finding = ResearchFinding(
            question_id=question["question_id"],
            finding=f"Web research conducted but compression failed: {e}",
            sources=[
                WebSearchResult(
                    url=r["url"],
                    title=r["title"],
                    description=r.get("description"),
                    content=r.get("content"),
                    source_metadata=r.get("source_metadata"),
                )
                for r in search_results[:5]
            ],
            confidence=0.3,
            gaps=["Compression failed - raw data available"],
            language_code=language_code,
        )
        return {"finding": finding, "research_findings": [finding]}


def create_web_researcher_subgraph() -> StateGraph:
    """Create web researcher agent subgraph.

    Flow:
    START -> generate_queries -> execute_searches -> scrape_pages -> compress_findings -> END
    """
    builder = StateGraph(ResearcherState)

    # Use web-optimized query generation
    builder.add_node("generate_queries", create_generate_queries("web"))
    builder.add_node("execute_searches", execute_searches)
    builder.add_node("scrape_pages", scrape_pages)
    builder.add_node("compress_findings", compress_findings)

    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "execute_searches")
    builder.add_edge("execute_searches", "scrape_pages")
    builder.add_edge("scrape_pages", "compress_findings")
    builder.add_edge("compress_findings", END)

    return builder.compile()


# Export compiled subgraph
web_researcher_subgraph = create_web_researcher_subgraph()
