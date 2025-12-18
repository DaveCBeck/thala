"""
Individual researcher agent subgraph.

Each researcher:
1. Generates search queries from the research question
2. Searches the web via Firecrawl
3. Scrapes relevant pages for full content
4. Compresses findings into structured output

Uses HAIKU for cost-effective research operations.
"""

import json
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from langchain_tools.firecrawl import web_search, scrape_url
from workflows.research.state import ResearcherState, ResearchFinding, WebSearchResult
from workflows.research.prompts import RESEARCHER_SYSTEM, COMPRESS_RESEARCH_SYSTEM, get_today_str
from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)

# Maximum searches per researcher
MAX_SEARCHES = 5
MAX_SCRAPES = 3


async def generate_queries(state: ResearcherState) -> dict[str, Any]:
    """Generate search queries for the research question."""
    question = state["question"]

    llm = get_llm(ModelTier.HAIKU)
    prompt = f"""Generate 2-3 search queries to research this question:

Question: {question['question']}
Context: {question.get('context', 'No additional context')}

Output as a JSON array of strings: ["query1", "query2", "query3"]

Make queries specific and likely to find authoritative sources.
"""

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        content = response.content.strip()

        # Extract JSON from response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        queries = json.loads(content)

        logger.debug(f"Generated {len(queries)} search queries for: {question['question'][:50]}...")
        return {"search_queries": queries}

    except Exception as e:
        logger.error(f"Failed to generate queries: {e}")
        # Fallback: use the question as the query
        return {"search_queries": [question["question"]]}


async def execute_searches(state: ResearcherState) -> dict[str, Any]:
    """Execute web searches using Firecrawl."""
    queries = state.get("search_queries", [])
    all_results = []

    for query in queries[:MAX_SEARCHES]:
        try:
            result = await web_search.ainvoke({"query": query, "limit": 5})
            results_list = result.get("results", [])

            for r in results_list:
                all_results.append(
                    WebSearchResult(
                        url=r.get("url", ""),
                        title=r.get("title", ""),
                        description=r.get("description"),
                        content=None,  # Will be filled by scraping
                    )
                )
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique_results.append(r)

    logger.info(f"Found {len(unique_results)} unique results from {len(queries)} queries")
    return {"search_results": unique_results[:10]}


async def scrape_pages(state: ResearcherState) -> dict[str, Any]:
    """Scrape top results for full content."""
    results = state.get("search_results", [])
    scraped = []
    updated_results = []

    # Scrape top N most relevant
    for i, result in enumerate(results[:MAX_SCRAPES]):
        try:
            response = await scrape_url.ainvoke({"url": result["url"]})
            content = response.get("markdown", "")

            # Truncate very long content
            if len(content) > 8000:
                content = content[:8000] + "\n\n[Content truncated...]"

            scraped.append(f"[{result['title']}]\nURL: {result['url']}\n\n{content}")

            # Update result with content
            updated_result = dict(result)
            updated_result["content"] = content
            updated_results.append(updated_result)

            logger.debug(f"Scraped {len(content)} chars from: {result['url']}")

        except Exception as e:
            logger.warning(f"Failed to scrape {result['url']}: {e}")
            updated_results.append(result)

    # Keep remaining results without scraping
    updated_results.extend(results[MAX_SCRAPES:])

    return {
        "scraped_content": scraped,
        "search_results": updated_results,
    }


async def compress_findings(state: ResearcherState) -> dict[str, Any]:
    """Compress research into structured finding."""
    question = state["question"]
    search_results = state.get("search_results", [])
    scraped = state.get("scraped_content", [])

    # Build raw research text
    if scraped:
        raw_research = "\n\n---\n\n".join(scraped)
    else:
        raw_research = "\n".join([
            f"- [{r['title']}]({r['url']}): {r.get('description', 'No description')}"
            for r in search_results
        ])

    prompt = COMPRESS_RESEARCH_SYSTEM.format(
        date=get_today_str(),
        question=question["question"],
        raw_research=raw_research[:15000],  # Limit context
    )

    llm = get_llm(ModelTier.SONNET)  # Use Sonnet for better compression

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        content = response.content.strip()

        # Extract JSON
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        finding_data = json.loads(content)

        finding = ResearchFinding(
            question_id=question["question_id"],
            finding=finding_data.get("finding", "No finding"),
            sources=[
                WebSearchResult(
                    url=s.get("url", ""),
                    title=s.get("title", ""),
                    description=s.get("relevance", ""),
                    content=None,
                )
                for s in finding_data.get("sources", [])
            ],
            confidence=float(finding_data.get("confidence", 0.5)),
            gaps=finding_data.get("gaps", []),
        )

        logger.info(
            f"Compressed finding for {question['question_id']}: "
            f"confidence={finding['confidence']:.2f}, gaps={len(finding['gaps'])}"
        )

        # Return as list for aggregation via add reducer in parent state
        return {"finding": finding, "research_findings": [finding]}

    except Exception as e:
        logger.error(f"Failed to compress findings: {e}")

        # Fallback finding
        finding = ResearchFinding(
            question_id=question["question_id"],
            finding=f"Research conducted but compression failed: {e}",
            sources=[
                WebSearchResult(
                    url=r["url"],
                    title=r["title"],
                    description=r.get("description"),
                    content=None,
                )
                for r in search_results[:5]
            ],
            confidence=0.3,
            gaps=["Compression failed - raw data available"],
        )
        return {"finding": finding, "research_findings": [finding]}


def create_researcher_subgraph() -> StateGraph:
    """Create researcher agent subgraph.

    Flow:
    START -> generate_queries -> execute_searches -> scrape_pages -> compress_findings -> END
    """
    builder = StateGraph(ResearcherState)

    builder.add_node("generate_queries", generate_queries)
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
researcher_subgraph = create_researcher_subgraph()
