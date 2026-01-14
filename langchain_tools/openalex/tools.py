"""LangChain tools for OpenAlex."""

import logging
from typing import Optional

from dotenv import load_dotenv
from langchain.tools import tool

from langchain_tools.utils import clamp_limit, output_dict
from workflows.shared.persistent_cache import get_cached, set_cached
from .client import _get_openalex
from .models import OpenAlexSearchOutput
from .parsing import _parse_work

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_TYPE = "openalex"
CACHE_TTL_DAYS = 30


@tool
async def openalex_search(
    query: str,
    limit: int = 10,
    min_citations: Optional[int] = None,
    from_year: Optional[int] = None,
    to_year: Optional[int] = None,
    language: Optional[str] = None,
) -> dict:
    """Search academic literature using OpenAlex.

    OpenAlex indexes 240M+ scholarly works including papers, books, datasets.
    Use this for finding academic/scientific sources and peer-reviewed research.

    This tool ALWAYS runs as part of research tasks to ensure academic sources
    are included alongside web search results.

    Args:
        query: What to search for in academic literature
        limit: Maximum number of results (default 10, max 50)
        min_citations: Minimum citation count filter (optional)
        from_year: Only include works from this year onwards (optional)
        to_year: Only include works up to this year (optional)
        language: ISO 639-1 language code (e.g., "es", "zh", "ja")

    Returns:
        Academic works with titles, abstracts, authors, citations, and DOIs.
    """
    cache_key = (
        f"search:{query}:{limit}:{min_citations}:{from_year}:{to_year}:{language}"
    )

    cached = get_cached(CACHE_TYPE, cache_key, ttl_days=CACHE_TTL_DAYS)
    if cached:
        logger.debug(f"Cache hit for OpenAlex search: {query}")
        return cached

    client = _get_openalex()
    limit = clamp_limit(limit, min_val=1, max_val=50)

    try:
        # Build filter string
        filters = []
        if min_citations is not None:
            filters.append(f"cited_by_count:>{min_citations}")
        if from_year is not None:
            filters.append(f"publication_year:>{from_year - 1}")
        if to_year is not None:
            filters.append(f"publication_year:<{to_year + 1}")
        if language is not None:
            filters.append(f"language:{language}")

        params = {
            "search": query,
            "per_page": limit,
            "sort": "relevance_score:desc",
        }
        if filters:
            params["filter"] = ",".join(filters)

        response = await client.get("/works", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for work in data.get("results", []):
            try:
                parsed = _parse_work(work)
                results.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse OpenAlex work: {e}")
                continue

        output = OpenAlexSearchOutput(
            query=query,
            total_results=data.get("meta", {}).get("count", len(results)),
            results=results,
        )

        logger.debug(
            f"OpenAlex search returned {len(results)} results for '{query}' "
            f"(total in index: {output.total_results})"
        )

        result_dict = output_dict(output)
        set_cached(CACHE_TYPE, cache_key, result_dict)
        return result_dict

    except Exception as e:
        logger.error(f"openalex_search failed for '{query}': {e}")
        return output_dict(
            OpenAlexSearchOutput(
                query=query,
                total_results=0,
                results=[],
            )
        )
