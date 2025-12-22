"""
Individual researcher agent subgraph.

Each researcher:
1. Generates search queries from the research question
2. Searches multiple sources in parallel (Firecrawl, Perplexity, OpenAlex, Books)
3. Scrapes relevant pages for full content (with TTL caching to avoid redundant scrapes)
4. Compresses findings into structured output

Uses HAIKU for cost-effective research operations.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any

import httpx
from cachetools import TTLCache
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
from workflows.shared.marker_client import MarkerClient

logger = logging.getLogger(__name__)

# Maximum searches per researcher
MAX_SEARCHES = 5
MAX_SCRAPES = 3
MAX_RESULTS_PER_SOURCE = 5  # Limit results from each search source

# Marker input directory for PDF processing
MARKER_INPUT_DIR = os.getenv(
    "MARKER_INPUT_DIR",
    "/home/dave/thala/services/marker/data/input"
)

# Cache for scraped URL content (1 hour TTL, max 200 items)
# This avoids re-scraping the same URL across different researchers or iterations
_scrape_cache: TTLCache = TTLCache(
    maxsize=int(os.getenv("SCRAPE_CACHE_SIZE", "200")),
    ttl=int(os.getenv("SCRAPE_CACHE_TTL", "3600")),  # 1 hour default
)


def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file."""
    return url.lower().rstrip('/').endswith('.pdf')


async def fetch_pdf_via_marker(url: str) -> str | None:
    """Download PDF and convert via Marker service.

    Returns markdown content or None if failed.
    """
    filename = f"{uuid.uuid4().hex}.pdf"
    input_path = os.path.join(MARKER_INPUT_DIR, filename)

    try:
        # Download PDF
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

        # Write to Marker input directory
        with open(input_path, "wb") as f:
            f.write(response.content)

        # Convert via Marker
        async with MarkerClient() as marker:
            result = await marker.convert(
                file_path=filename,
                quality="fast",  # Fast preset for research scraping
            )
            return result.markdown

    except Exception as e:
        logger.warning(f"Marker PDF processing failed for {url}: {e}")
        return None

    finally:
        # Cleanup temp file
        try:
            os.remove(input_path)
        except OSError:
            pass


async def _scrape_single_url(
    result: WebSearchResult,
    index: int,
) -> tuple[int, str | None, WebSearchResult]:
    """Scrape a single URL and return the result.

    This helper function enables parallel scraping via asyncio.gather().
    Uses a TTL cache to avoid re-scraping the same URL across researchers or iterations.

    Args:
        result: WebSearchResult containing URL and metadata
        index: Original index in results list (for preserving order)

    Returns:
        Tuple of (index, scraped_content_str, updated_result)
        - index: Original position for deterministic ordering
        - scraped_content_str: Formatted content string or None on failure
        - updated_result: Result dict with content field added
    """
    url = result["url"]
    updated_result = dict(result)
    updated_result["_index"] = index

    # Check cache first
    if url in _scrape_cache:
        content = _scrape_cache[url]
        logger.debug(f"Cache hit for: {url} ({len(content)} chars)")

        # Format scraped content string
        content_str = f"[{result['title']}]\nURL: {result['url']}\n\n{content}"
        updated_result["content"] = content

        return (index, content_str, updated_result)

    try:
        # Route PDFs to Marker instead of Firecrawl
        if is_pdf_url(url):
            logger.info(f"Processing PDF via Marker: {url}")
            content = await fetch_pdf_via_marker(url)
            if not content:
                # Fallback to Firecrawl if Marker fails
                logger.info(f"Marker failed, falling back to Firecrawl: {url}")
                response = await scrape_url.ainvoke({"url": url})
                content = response.get("markdown", "")
        else:
            response = await scrape_url.ainvoke({"url": url})
            content = response.get("markdown", "")

        # Truncate very long content
        if len(content) > 8000:
            content = content[:8000] + "\n\n[Content truncated...]"

        # Store in cache for future use
        _scrape_cache[url] = content
        logger.debug(f"Scraped and cached {len(content)} chars from: {url}")

        # Format scraped content string
        content_str = f"[{result['title']}]\nURL: {result['url']}\n\n{content}"

        # Update result with content
        updated_result["content"] = content

        return (index, content_str, updated_result)

    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return (index, None, updated_result)


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
    """Scrape top results for full content in parallel.

    Routes PDF URLs to local Marker service instead of Firecrawl to save API costs.
    Uses asyncio.gather() to scrape multiple URLs concurrently for improved performance.
    Results are cached with 1-hour TTL to avoid redundant scrapes across researchers.
    """
    results = state.get("search_results", [])

    if not results:
        return {"scraped_content": [], "search_results": []}

    # Check how many URLs are already cached
    urls_to_scrape = [r["url"] for r in results[:MAX_SCRAPES]]
    cached_count = sum(1 for url in urls_to_scrape if url in _scrape_cache)
    logger.info(
        f"Scraping {len(urls_to_scrape)} URLs ({cached_count} cached, "
        f"{len(urls_to_scrape) - cached_count} new) - cache size: {len(_scrape_cache)}"
    )

    # Create scraping tasks for parallel execution
    scraping_tasks = [
        _scrape_single_url(result, i)
        for i, result in enumerate(results[:MAX_SCRAPES])
    ]

    # Execute all scrapes concurrently
    task_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)

    # Process results
    scraped = []
    updated_results = []

    for task_result in task_results:
        if isinstance(task_result, Exception):
            logger.error(f"Scraping task failed with exception: {task_result}")
            continue

        idx, content_str, updated_result = task_result

        if content_str:
            scraped.append(content_str)

        updated_results.append(updated_result)

    # Sort by original index to maintain deterministic order
    updated_results.sort(key=lambda x: x.get("_index", 0))

    # Remove temporary index field
    for r in updated_results:
        r.pop("_index", None)

    # Keep remaining results without scraping
    updated_results.extend(results[MAX_SCRAPES:])

    logger.info(f"Scraped {len(scraped)} URLs successfully in parallel")

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
