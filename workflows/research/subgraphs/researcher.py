"""
Individual researcher agent subgraph.

Each researcher:
1. Generates search queries from the research question
2. Searches multiple sources in parallel (Firecrawl, Perplexity, OpenAlex, Books)
3. Scrapes relevant pages for full content
4. Compresses findings into structured output

Uses HAIKU for cost-effective research operations.
"""

import asyncio
import json
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from langchain_tools.firecrawl import web_search, scrape_url
from langchain_tools.perplexity import perplexity_search
from langchain_tools.openalex import openalex_search
from langchain_tools.book_search import book_search
from workflows.research.state import (
    ResearcherState,
    ResearchFinding,
    WebSearchResult,
    SearchQueries,
    QueryValidationBatch,
)
from workflows.research.prompts import (
    RESEARCHER_SYSTEM,
    COMPRESS_RESEARCH_SYSTEM_CACHED,
    COMPRESS_RESEARCH_USER_TEMPLATE,
    get_today_str,
)
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache

logger = logging.getLogger(__name__)

# Maximum searches per researcher
MAX_SEARCHES = 5
MAX_SCRAPES = 3
MAX_RESULTS_PER_SOURCE = 5  # Limit results from each search source


async def validate_queries(
    queries: list[str],
    research_question: str,
    research_brief: dict | None = None,
    draft_notes: str | None = None,
) -> list[str]:
    """
    Validate queries using LLM to ensure they're relevant to the research.

    Args:
        queries: Generated search queries to validate
        research_question: The original research question
        research_brief: Optional research brief for context
        draft_notes: Optional current draft/notes for context

    Returns:
        List of validated queries that are relevant to the research
    """
    if not queries:
        return []

    llm = get_llm(ModelTier.HAIKU)
    structured_llm = llm.with_structured_output(QueryValidationBatch)

    # Build context
    context_parts = [f"Research Question: {research_question}"]
    if research_brief:
        context_parts.append(f"Topic: {research_brief.get('topic', '')}")
        if research_brief.get('objectives'):
            context_parts.append(f"Objectives: {', '.join(research_brief['objectives'][:3])}")
    if draft_notes:
        context_parts.append(f"Current Notes: {draft_notes[:500]}...")

    context = "\n".join(context_parts)
    queries_list = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))

    prompt = f"""Validate whether these search queries are relevant to the research task.

{context}

Proposed Search Queries:
{queries_list}

For each query, determine if it's actually relevant to the research question above.
Reject queries that:
- Contain system metadata (iteration counts, percentages, internal state)
- Are about completely unrelated topics
- Are too vague or generic to be useful

Accept queries that would help find information about the research topic.
"""

    try:
        result: QueryValidationBatch = await structured_llm.ainvoke(
            [{"role": "user", "content": prompt}]
        )

        valid_queries = []
        for query, validation in zip(queries, result.validations):
            if validation.is_relevant:
                valid_queries.append(query)
            else:
                logger.warning(f"Query rejected: {query[:50]}... Reason: {validation.reason}")

        return valid_queries

    except Exception as e:
        logger.warning(f"Query validation failed: {e}, accepting all queries")
        return queries  # Fail open


async def generate_queries(state: ResearcherState) -> dict[str, Any]:
    """Generate search queries using structured output with validation."""
    question = state["question"]

    llm = get_llm(ModelTier.HAIKU)
    structured_llm = llm.with_structured_output(SearchQueries)

    prompt = f"""Generate 2-3 search queries to research this question:

Question: {question['question']}

Make queries specific and likely to find authoritative sources.
Focus only on the research topic - do not include any system metadata.
"""

    try:
        result: SearchQueries = await structured_llm.ainvoke([{"role": "user", "content": prompt}])

        # Validate queries are relevant
        valid_queries = await validate_queries(
            queries=result.queries,
            research_question=question['question'],
            research_brief=question.get('brief'),
            draft_notes=question.get('context'),
        )

        if not valid_queries:
            logger.warning("All queries invalid, using fallback")
            valid_queries = [question["question"]]

        logger.debug(f"Generated {len(valid_queries)} valid queries for: {question['question'][:50]}...")
        return {"search_queries": valid_queries}

    except Exception as e:
        logger.error(f"Failed to generate queries: {e}")
        # Fallback: use the question as the query
        return {"search_queries": [question["question"]]}


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
                    source_metadata=None,  # No structured metadata for web sources
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
                    description=r.get("snippet"),  # Perplexity uses 'snippet'
                    content=None,
                    source_metadata=None,  # No structured metadata for web sources
                )
            )
    except Exception as e:
        logger.warning(f"Perplexity search failed for '{query}': {e}")
    return results


async def _search_openalex(query: str) -> list[WebSearchResult]:
    """Search using OpenAlex (academic sources)."""
    results = []
    try:
        result = await openalex_search.ainvoke({"query": query, "limit": MAX_RESULTS_PER_SOURCE})
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
                    url=r.get("url", ""),  # Now oa_url or DOI (for scraping)
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


async def _search_books(query: str) -> list[WebSearchResult]:
    """Search for books."""
    results = []
    try:
        result = await book_search.ainvoke({"query": query, "limit": MAX_RESULTS_PER_SOURCE})
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
    """Execute web searches using multiple sources in parallel.

    Sources:
    - Firecrawl: General web search
    - Perplexity: AI-powered web search with synthesis
    - OpenAlex: Academic/scholarly literature
    - Books: Book search
    """
    queries = state.get("search_queries", [])
    all_results = []

    for query in queries[:MAX_SEARCHES]:
        # Run all four sources in parallel for each query
        search_tasks = [
            _search_firecrawl(query),
            _search_perplexity(query),
            _search_openalex(query),
            _search_books(query),
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

    logger.info(
        f"Found {len(unique_results)} unique results from {len(queries)} queries "
        f"across Firecrawl/Perplexity/OpenAlex/Books"
    )
    return {"search_results": unique_results[:15]}


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

    # Build dynamic user prompt
    user_prompt = COMPRESS_RESEARCH_USER_TEMPLATE.format(
        date=get_today_str(),
        question=question["question"],
        raw_research=raw_research[:15000],  # Limit context
    )

    llm = get_llm(ModelTier.SONNET)  # Use Sonnet for better compression

    try:
        # Use cached system prompt for 90% cost reduction on repeated calls
        response = await invoke_with_cache(
            llm,
            system_prompt=COMPRESS_RESEARCH_SYSTEM_CACHED,  # ~400 tokens, cached
            user_prompt=user_prompt,  # Dynamic content
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
        )

        logger.info(
            f"Compressed finding for {question['question_id']}: "
            f"confidence={finding['confidence']:.2f}, gaps={len(finding['gaps'])}"
        )

        # Return as list for aggregation via add reducer in parent state
        return {"finding": finding, "research_findings": [finding]}

    except Exception as e:
        logger.error(f"Failed to compress findings: {e}")

        # Fallback finding - preserve all original data including source_metadata
        finding = ResearchFinding(
            question_id=question["question_id"],
            finding=f"Research conducted but compression failed: {e}",
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
