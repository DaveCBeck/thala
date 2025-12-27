"""
Book researcher subgraph.

Specialized researcher for book content using:
- book_search: Book search API (e.g., Library Genesis, Open Library)

Uses the book-specific compression prompt that emphasizes:
- Author credentials and expertise
- Publisher reputation (academic vs trade vs self-published)
- Edition currency and book's contribution to the field

Default: 1 book per research question.
"""

import asyncio
import json
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from langchain_tools.book_search import book_search
from workflows.research.state import (
    ResearcherState,
    ResearchFinding,
    WebSearchResult,
)
from workflows.research.prompts import (
    COMPRESS_BOOK_RESEARCH_SYSTEM,
    COMPRESS_RESEARCH_USER_TEMPLATE,
    get_today_str,
)
from workflows.research.prompts.translator import get_translated_prompt
from workflows.research.subgraphs.researcher_base import (
    create_generate_queries,
    scrape_pages as base_scrape_pages,
)
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache

logger = logging.getLogger(__name__)

# Book researcher constants
MAX_SEARCHES = 2          # Focused book queries
MAX_SCRAPES = 1           # Default 1 book (per requirements)
MAX_RESULTS_PER_SEARCH = 3


async def _search_books(query: str) -> list[WebSearchResult]:
    """Search for books using book_search API."""
    results = []
    try:
        result = await book_search.ainvoke({"query": query, "limit": MAX_RESULTS_PER_SEARCH})
        for r in result.get("results", []):
            # Build description from authors + format + size
            description = f"{r.get('authors', 'Unknown')} · {r.get('format', '').upper()} · {r.get('size', '')}"
            if r.get("abstract"):
                description += f"\n{r['abstract']}"

            results.append(
                WebSearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    description=description,
                    content=None,
                    source_metadata={
                        "source_type": "book",
                        "md5": r.get("md5"),
                        "authors": r.get("authors"),
                        "publisher": r.get("publisher"),
                        "language": r.get("language"),
                        "format": r.get("format"),
                        "size": r.get("size"),
                        "abstract": r.get("abstract"),
                    },
                )
            )
    except Exception as e:
        logger.warning(f"Book search failed for '{query}': {e}")
    return results


async def execute_searches(state: ResearcherState) -> dict[str, Any]:
    """Execute book searches.

    Searches for books using the book_search API.
    Prioritizes PDF format for easier scraping.
    """
    queries = state.get("search_queries", [])
    all_results = []

    for query in queries[:MAX_SEARCHES]:
        results = await _search_books(query)
        all_results.extend(results)

    # Deduplicate by MD5 or URL
    seen_keys = set()
    unique_results = []
    for r in all_results:
        md5 = r.get("source_metadata", {}).get("md5") if r.get("source_metadata") else None
        key = md5 or r["url"]
        if key and key not in seen_keys:
            seen_keys.add(key)
            unique_results.append(r)

    # Prioritize PDF format over other formats (easier to process via Marker)
    def format_priority(result):
        fmt = result.get("source_metadata", {}).get("format", "").lower() if result.get("source_metadata") else ""
        if fmt == "pdf":
            return 0
        elif fmt == "epub":
            return 1
        else:
            return 2

    unique_results.sort(key=format_priority)

    logger.info(
        f"Book researcher: Found {len(unique_results)} unique results from {len(queries)} queries "
        f"via book_search"
    )
    return {"search_results": unique_results[:5]}


async def scrape_pages(state: ResearcherState) -> dict[str, Any]:
    """Scrape top book for full content.

    Books are typically PDFs that get processed via Marker.
    """
    return await base_scrape_pages(state, max_scrapes=MAX_SCRAPES)


async def compress_findings(state: ResearcherState) -> dict[str, Any]:
    """Compress research into structured finding using book-specific prompt.

    Emphasizes:
    - Author credentials and field expertise
    - Publisher reputation
    - Book's theoretical framework and contribution
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

    # Get language-appropriate prompts
    if language_config and language_config["code"] != "en":
        system_prompt = await get_translated_prompt(
            COMPRESS_BOOK_RESEARCH_SYSTEM,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="compress_book_research_system",
        )
        user_template = await get_translated_prompt(
            COMPRESS_RESEARCH_USER_TEMPLATE,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="compress_research_user",
        )
    else:
        system_prompt = COMPRESS_BOOK_RESEARCH_SYSTEM
        user_template = COMPRESS_RESEARCH_USER_TEMPLATE

    # Build dynamic user prompt
    user_prompt = user_template.format(
        date=get_today_str(),
        question=question["question"],
        raw_research=raw_research[:15000],  # Limit context
    )

    llm = get_llm(ModelTier.SONNET)  # Use Sonnet for better compression

    try:
        # Use cached system prompt for 90% cost reduction on repeated calls
        response = await invoke_with_cache(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        content = content.strip()

        # Extract JSON
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        finding_data = json.loads(content)

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
                    content=original.get("content"),  # Preserve scraped content
                    source_metadata=original.get("source_metadata"),  # Preserve book metadata
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

        logger.info(
            f"Book researcher compressed finding for {question['question_id']}: "
            f"confidence={finding['confidence']:.2f}, gaps={len(finding['gaps'])}"
        )

        # Return as list for aggregation via add reducer in parent state
        return {"finding": finding, "research_findings": [finding]}

    except Exception as e:
        logger.error(f"Failed to compress book findings: {e}")

        # Fallback finding - preserve all original data including source_metadata
        finding = ResearchFinding(
            question_id=question["question_id"],
            finding=f"Book research conducted but compression failed: {e}",
            sources=[
                WebSearchResult(
                    url=r["url"],
                    title=r["title"],
                    description=r.get("description"),
                    content=r.get("content"),
                    source_metadata=r.get("source_metadata"),  # Preserve book metadata
                )
                for r in search_results[:5]
            ],
            confidence=0.3,
            gaps=["Compression failed - raw data available"],
            language_code=language_code,
        )
        return {"finding": finding, "research_findings": [finding]}


def create_book_researcher_subgraph() -> StateGraph:
    """Create book researcher agent subgraph.

    Flow:
    START -> generate_queries -> execute_searches -> scrape_pages -> compress_findings -> END
    """
    builder = StateGraph(ResearcherState)

    # Use book-optimized query generation for book_search
    builder.add_node("generate_queries", create_generate_queries("book"))
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
book_researcher_subgraph = create_book_researcher_subgraph()
