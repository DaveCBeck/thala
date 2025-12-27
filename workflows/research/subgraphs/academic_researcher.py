"""
Academic researcher subgraph.

Specialized researcher for scholarly/academic content using:
- OpenAlex: Academic/scholarly literature database

Uses the academic-specific compression prompt that emphasizes:
- Peer-review status and journal reputation
- Citation counts and evidence strength
- Methodology quality and study limitations

Default: 3 articles per research question.
"""

import asyncio
import json
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from langchain_tools.openalex import openalex_search
from workflows.research.state import (
    ResearcherState,
    ResearchFinding,
    WebSearchResult,
)
from workflows.research.prompts import (
    COMPRESS_ACADEMIC_RESEARCH_SYSTEM,
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

# Academic researcher constants
MAX_SEARCHES = 3          # Fewer but more focused queries
MAX_SCRAPES = 3           # Default 3 articles (per requirements)
MAX_RESULTS_PER_SEARCH = 5


async def _search_openalex(query: str) -> list[WebSearchResult]:
    """Search using OpenAlex (academic sources)."""
    results = []
    try:
        result = await openalex_search.ainvoke({"query": query, "limit": MAX_RESULTS_PER_SEARCH})
        for r in result.get("results", []):
            # Format authors for description
            authors = r.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            # Build description from abstract + metadata
            abstract = r.get("abstract", "") or ""
            description = f"{author_str}. " if author_str else ""
            if r.get("publication_date"):
                description += f"({r['publication_date'][:4]}). "
            if r.get("source_name"):
                description += f"{r['source_name']}. "
            description += f"Cited by {r.get('cited_by_count', 0)}. "
            if abstract:
                description += abstract[:200] + "..." if len(abstract) > 200 else abstract

            results.append(
                WebSearchResult(
                    url=r.get("url", ""),  # oa_url or DOI (for scraping)
                    title=r.get("title", ""),
                    description=description,
                    content=None,
                    source_metadata={  # Structured data for citation processor
                        "source_type": "openalex",
                        "doi": r.get("doi"),
                        "oa_url": r.get("oa_url"),
                        "authors": r.get("authors", []),
                        "publication_date": r.get("publication_date"),
                        "cited_by_count": r.get("cited_by_count", 0),
                        "source_name": r.get("source_name"),
                        "primary_topic": r.get("primary_topic"),
                        "abstract": r.get("abstract"),
                        "is_oa": r.get("is_oa", False),
                        "oa_status": r.get("oa_status"),
                    },
                )
            )
    except Exception as e:
        logger.warning(f"OpenAlex search failed for '{query}': {e}")
    return results


async def execute_searches(state: ResearcherState) -> dict[str, Any]:
    """Execute academic searches using OpenAlex.

    Searches OpenAlex for peer-reviewed academic literature.
    """
    queries = state.get("search_queries", [])
    all_results = []

    for query in queries[:MAX_SEARCHES]:
        results = await _search_openalex(query)
        all_results.extend(results)

    # Deduplicate by URL (or DOI if available)
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r["url"]
        doi = r.get("source_metadata", {}).get("doi") if r.get("source_metadata") else None
        key = doi or url
        if key and key not in seen_urls:
            seen_urls.add(key)
            unique_results.append(r)

    # Sort by citation count (higher citations = more established)
    unique_results.sort(
        key=lambda x: x.get("source_metadata", {}).get("cited_by_count", 0)
        if x.get("source_metadata") else 0,
        reverse=True
    )

    logger.info(
        f"Academic researcher: Found {len(unique_results)} unique results from {len(queries)} queries "
        f"via OpenAlex"
    )
    return {"search_results": unique_results[:10]}


async def scrape_pages(state: ResearcherState) -> dict[str, Any]:
    """Scrape top academic articles for full content."""
    return await base_scrape_pages(state, max_scrapes=MAX_SCRAPES)


async def compress_findings(state: ResearcherState) -> dict[str, Any]:
    """Compress research into structured finding using academic-specific prompt.

    Emphasizes:
    - Peer-review status and journal reputation
    - Citation counts and methodology quality
    - Evidence strength (meta-analysis > RCT > observational)
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
            COMPRESS_ACADEMIC_RESEARCH_SYSTEM,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="compress_academic_research_system",
        )
        user_template = await get_translated_prompt(
            COMPRESS_RESEARCH_USER_TEMPLATE,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="compress_research_user",
        )
    else:
        system_prompt = COMPRESS_ACADEMIC_RESEARCH_SYSTEM
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
                    source_metadata=original.get("source_metadata"),  # Preserve OpenAlex metadata
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
            f"Academic researcher compressed finding for {question['question_id']}: "
            f"confidence={finding['confidence']:.2f}, gaps={len(finding['gaps'])}"
        )

        # Return as list for aggregation via add reducer in parent state
        return {"finding": finding, "research_findings": [finding]}

    except Exception as e:
        logger.error(f"Failed to compress academic findings: {e}")

        # Fallback finding - preserve all original data including source_metadata
        finding = ResearchFinding(
            question_id=question["question_id"],
            finding=f"Academic research conducted but compression failed: {e}",
            sources=[
                WebSearchResult(
                    url=r["url"],
                    title=r["title"],
                    description=r.get("description"),
                    content=r.get("content"),
                    source_metadata=r.get("source_metadata"),  # Preserve OpenAlex metadata
                )
                for r in search_results[:5]
            ],
            confidence=0.3,
            gaps=["Compression failed - raw data available"],
            language_code=language_code,
        )
        return {"finding": finding, "research_findings": [finding]}


def create_academic_researcher_subgraph() -> StateGraph:
    """Create academic researcher agent subgraph.

    Flow:
    START -> generate_queries -> execute_searches -> scrape_pages -> compress_findings -> END
    """
    builder = StateGraph(ResearcherState)

    # Use academic-optimized query generation for OpenAlex
    builder.add_node("generate_queries", create_generate_queries("academic"))
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
academic_researcher_subgraph = create_academic_researcher_subgraph()
