"""
OpenAlex academic search tools for LangChain.

OpenAlex is a free, open catalog of 240M+ scholarly works.
Provides: openalex_search
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client Management (lazy singleton)
# ---------------------------------------------------------------------------

_openalex_client = None


def _get_openalex():
    """Get OpenAlex httpx client (lazy init)."""
    global _openalex_client
    if _openalex_client is None:
        import httpx

        # OpenAlex recommends providing email for polite pool (faster rate limits)
        email = os.environ.get("OPENALEX_EMAIL", "")
        headers = {}
        if email:
            headers["User-Agent"] = f"mailto:{email}"

        _openalex_client = httpx.AsyncClient(
            base_url="https://api.openalex.org",
            headers=headers,
            timeout=30.0,
        )
    return _openalex_client


# ---------------------------------------------------------------------------
# Output Models
# ---------------------------------------------------------------------------


class OpenAlexAuthor(BaseModel):
    """Author information from OpenAlex."""

    name: str
    institution: Optional[str] = None


class OpenAlexWork(BaseModel):
    """Individual academic work from OpenAlex."""

    title: str
    url: str  # oa_url if available, else DOI (preferred for scraping)
    doi: Optional[str] = None  # Always keep DOI for citations
    oa_url: Optional[str] = None  # Open access URL for full text
    abstract: Optional[str] = None
    authors: list[OpenAlexAuthor] = Field(default_factory=list)
    publication_date: Optional[str] = None
    cited_by_count: int = 0
    primary_topic: Optional[str] = None
    source_name: Optional[str] = None  # Journal/venue name
    is_oa: bool = False  # Whether work is open access
    oa_status: Optional[str] = None  # gold, green, hybrid, bronze, closed


class OpenAlexSearchOutput(BaseModel):
    """Output schema for openalex_search tool."""

    query: str
    total_results: int
    results: list[OpenAlexWork]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""

    # Build word->position mapping
    words_with_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words_with_positions.append((pos, word))

    # Sort by position and join
    words_with_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words_with_positions)


def _parse_work(work: dict) -> OpenAlexWork:
    """Parse OpenAlex work response into our model."""
    # Get open access info
    oa_info = work.get("open_access", {})
    oa_url = oa_info.get("oa_url")  # Best OA URL for full text
    is_oa = oa_info.get("is_oa", False)
    oa_status = oa_info.get("oa_status")

    # Get DOI (always keep for citations)
    doi = work.get("doi")

    # Prefer oa_url for scraping, fallback to DOI, then OpenAlex ID
    url = oa_url or doi or work.get("id", "")

    # Parse authors (limit to first 5)
    authors = []
    for authorship in work.get("authorships", [])[:5]:
        author = authorship.get("author", {})
        institutions = authorship.get("institutions", [])
        institution_name = institutions[0].get("display_name") if institutions else None

        authors.append(
            OpenAlexAuthor(
                name=author.get("display_name", "Unknown"),
                institution=institution_name,
            )
        )

    # Get primary topic
    primary_topic = None
    topic_data = work.get("primary_topic")
    if topic_data:
        primary_topic = topic_data.get("display_name")

    # Get source/journal name
    source_name = None
    primary_location = work.get("primary_location", {})
    if primary_location:
        source = primary_location.get("source", {})
        if source:
            source_name = source.get("display_name")

    return OpenAlexWork(
        title=work.get("title") or work.get("display_name") or "Untitled",
        url=url,
        doi=doi,
        oa_url=oa_url,
        abstract=_reconstruct_abstract(work.get("abstract_inverted_index", {})),
        authors=authors,
        publication_date=work.get("publication_date"),
        cited_by_count=work.get("cited_by_count", 0),
        primary_topic=primary_topic,
        source_name=source_name,
        is_oa=is_oa,
        oa_status=oa_status,
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


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
    client = _get_openalex()
    limit = min(max(1, limit), 50)

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
            f"openalex_search returned {len(results)} results for: {query} "
            f"(total in index: {output.total_results})"
        )
        return output.model_dump(mode="json")

    except Exception as e:
        logger.error(f"openalex_search failed: {e}")
        return OpenAlexSearchOutput(
            query=query,
            total_results=0,
            results=[],
        ).model_dump(mode="json")
