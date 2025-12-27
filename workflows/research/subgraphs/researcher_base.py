"""
Shared utilities for specialized researcher subgraphs.

Contains:
- TTL scrape cache (module-level singleton shared across all researchers)
- PDF processing via Marker
- URL scraping helper
- Query validation
- Query generation (identical for all researcher types)
"""

import asyncio
import logging
import os
import uuid
from typing import Any

import httpx
from cachetools import TTLCache

from langchain_tools.firecrawl import scrape_url
from workflows.research.state import (
    ResearcherState,
    WebSearchResult,
    SearchQueries,
    QueryValidationBatch,
)
from workflows.research.prompts import (
    GENERATE_WEB_QUERIES_SYSTEM,
    GENERATE_ACADEMIC_QUERIES_SYSTEM,
    GENERATE_BOOK_QUERIES_SYSTEM,
)
from workflows.shared.llm_utils import ModelTier, get_llm
from workflows.shared.marker_client import MarkerClient

logger = logging.getLogger(__name__)

# =============================================================================
# Shared TTL Cache (Module-Level Singleton)
# =============================================================================

# Cache for scraped URL content (1 hour TTL, max 200 items)
# Shared across ALL researcher types to avoid redundant scrapes
_scrape_cache: TTLCache = TTLCache(
    maxsize=int(os.getenv("SCRAPE_CACHE_SIZE", "200")),
    ttl=int(os.getenv("SCRAPE_CACHE_TTL", "3600")),  # 1 hour default
)


def get_scrape_cache() -> TTLCache:
    """Get the shared scrape cache instance."""
    return _scrape_cache


# =============================================================================
# Marker Input Directory
# =============================================================================

MARKER_INPUT_DIR = os.getenv(
    "MARKER_INPUT_DIR",
    "/home/dave/thala/services/marker/data/input"
)


# =============================================================================
# PDF Processing
# =============================================================================


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


# =============================================================================
# URL Scraping
# =============================================================================


async def scrape_single_url(
    result: WebSearchResult,
    index: int,
    max_content_length: int = 8000,
) -> tuple[int, str | None, WebSearchResult]:
    """Scrape a single URL and return the result.

    This helper function enables parallel scraping via asyncio.gather().
    Uses a TTL cache to avoid re-scraping the same URL across researchers or iterations.

    Args:
        result: WebSearchResult containing URL and metadata
        index: Original index in results list (for preserving order)
        max_content_length: Maximum content length before truncation

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
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n\n[Content truncated...]"

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


# =============================================================================
# Query Validation
# =============================================================================


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


# =============================================================================
# Query Generation (shared node function)
# =============================================================================

# Map researcher types to their specialized prompts
RESEARCHER_QUERY_PROMPTS = {
    "web": GENERATE_WEB_QUERIES_SYSTEM,
    "academic": GENERATE_ACADEMIC_QUERIES_SYSTEM,
    "book": GENERATE_BOOK_QUERIES_SYSTEM,
}


def create_generate_queries(researcher_type: str = "web"):
    """Create a generate_queries node function for a specific researcher type.

    Args:
        researcher_type: One of "web", "academic", "book"

    Returns:
        Async function that generates queries optimized for the researcher type.
    """

    async def generate_queries(state: ResearcherState) -> dict[str, Any]:
        """Generate search queries using structured output with validation.

        Uses researcher-type-specific prompts to optimize queries for:
        - Web: General search engines (Firecrawl, Perplexity)
        - Academic: OpenAlex peer-reviewed literature
        - Book: book_search databases

        If language_config is set, generates queries in the target language
        for better search results in that language.
        """
        question = state["question"]
        language_config = state.get("language_config")

        llm = get_llm(ModelTier.HAIKU)
        structured_llm = llm.with_structured_output(SearchQueries)

        # Get researcher-specific base prompt
        base_prompt = RESEARCHER_QUERY_PROMPTS.get(researcher_type, GENERATE_WEB_QUERIES_SYSTEM)

        # Build language-aware prompt
        if language_config and language_config["code"] != "en":
            lang_name = language_config["name"]
            prompt = f"""{base_prompt}

Generate queries in {lang_name} to find {lang_name}-language sources.
Write queries naturally in {lang_name}.

Question: {question['question']}
"""
        else:
            prompt = f"""{base_prompt}

Question: {question['question']}
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

            lang_info = f" ({language_config['code']})" if language_config else ""
            logger.debug(
                f"Generated {len(valid_queries)} {researcher_type} queries{lang_info} "
                f"for: {question['question'][:50]}..."
            )
            return {"search_queries": valid_queries}

        except Exception as e:
            logger.error(f"Failed to generate {researcher_type} queries: {e}")
            # Fallback: use the question as the query
            return {"search_queries": [question["question"]]}

    return generate_queries


# Backwards compatibility: default web query generator
async def generate_queries(state: ResearcherState) -> dict[str, Any]:
    """Generate search queries (default: web-optimized).

    For specialized query generation, use create_generate_queries(researcher_type).
    """
    generator = create_generate_queries("web")
    return await generator(state)


# =============================================================================
# Shared Scrape Pages Function
# =============================================================================


async def scrape_pages(
    state: ResearcherState,
    max_scrapes: int,
) -> dict[str, Any]:
    """Scrape top results for full content in parallel.

    Routes PDF URLs to local Marker service instead of Firecrawl to save API costs.
    Uses asyncio.gather() to scrape multiple URLs concurrently for improved performance.
    Results are cached with 1-hour TTL to avoid redundant scrapes across researchers.

    Args:
        state: The researcher state containing search results
        max_scrapes: Maximum number of URLs to scrape
    """
    results = state.get("search_results", [])

    if not results:
        return {"scraped_content": [], "search_results": []}

    # Check how many URLs are already cached
    urls_to_scrape = [r["url"] for r in results[:max_scrapes]]
    cached_count = sum(1 for url in urls_to_scrape if url in _scrape_cache)
    logger.info(
        f"Scraping {len(urls_to_scrape)} URLs ({cached_count} cached, "
        f"{len(urls_to_scrape) - cached_count} new) - cache size: {len(_scrape_cache)}"
    )

    # Create scraping tasks for parallel execution
    scraping_tasks = [
        scrape_single_url(result, i)
        for i, result in enumerate(results[:max_scrapes])
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
    updated_results.extend(results[max_scrapes:])

    logger.info(f"Scraped {len(scraped)} URLs successfully in parallel")

    return {
        "scraped_content": scraped,
        "search_results": updated_results,
    }
