"""
OpenAlex academic search tools for LangChain.

OpenAlex is a free, open catalog of 240M+ scholarly works.
Provides: openalex_search
"""

import logging
import os
from typing import Literal, Optional

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


class OpenAlexCitationResult(BaseModel):
    """Output schema for citation retrieval."""

    source_doi: str
    direction: Literal["forward", "backward"]
    total_count: int
    results: list[OpenAlexWork]


class OpenAlexAuthorWorksResult(BaseModel):
    """Output schema for author works retrieval."""

    author_id: str
    author_name: str
    total_works: int
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


# ---------------------------------------------------------------------------
# Citation and Author Functions (for direct workflow use)
# ---------------------------------------------------------------------------


async def get_forward_citations(
    work_id: str,
    limit: int = 50,
    min_citations: Optional[int] = None,
    from_year: Optional[int] = None,
) -> OpenAlexCitationResult:
    """Get works that cite the given work (forward citations).

    Uses OpenAlex filter: cites:{work_id} to find all works that reference
    this work in their bibliography.

    Args:
        work_id: DOI (with or without https://doi.org/) or OpenAlex ID
        limit: Maximum results (default 50, max 200)
        min_citations: Only include citing works with at least this many citations
        from_year: Only include citing works from this year onwards

    Returns:
        OpenAlexCitationResult with citing works sorted by citation count
    """
    client = _get_openalex()
    limit = min(max(1, limit), 200)

    # Normalize work_id (remove https://doi.org/ prefix if present)
    work_id_clean = work_id.replace("https://doi.org/", "").replace("http://doi.org/", "")

    try:
        # Build filter string
        filters = [f"cites:{work_id_clean}"]
        if min_citations is not None:
            filters.append(f"cited_by_count:>{min_citations}")
        if from_year is not None:
            filters.append(f"publication_year:>{from_year - 1}")

        params = {
            "filter": ",".join(filters),
            "per_page": limit,
            "sort": "cited_by_count:desc",
        }

        response = await client.get("/works", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for work in data.get("results", []):
            try:
                parsed = _parse_work(work)
                results.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse forward citation work: {e}")
                continue

        total_count = data.get("meta", {}).get("count", len(results))

        logger.debug(
            f"get_forward_citations returned {len(results)} results for {work_id_clean} "
            f"(total: {total_count})"
        )

        return OpenAlexCitationResult(
            source_doi=work_id_clean,
            direction="forward",
            total_count=total_count,
            results=results,
        )

    except Exception as e:
        logger.error(f"get_forward_citations failed for {work_id_clean}: {e}")
        return OpenAlexCitationResult(
            source_doi=work_id_clean,
            direction="forward",
            total_count=0,
            results=[],
        )


async def get_backward_citations(
    work_id: str,
    limit: int = 50,
) -> OpenAlexCitationResult:
    """Get works cited by the given work (backward citations/references).

    First fetches the work to get referenced_works list, then fetches full
    metadata for each referenced work using pipe-delimited filter.

    Args:
        work_id: DOI (with or without https://doi.org/) or OpenAlex ID
        limit: Maximum results (default 50, max 200)

    Returns:
        OpenAlexCitationResult with referenced works
    """
    client = _get_openalex()
    limit = min(max(1, limit), 200)

    # Normalize work_id
    work_id_clean = work_id.replace("https://doi.org/", "").replace("http://doi.org/", "")

    try:
        # First, fetch the source work to get referenced_works
        # Handle both DOI and OpenAlex ID formats
        if work_id_clean.startswith("https://openalex.org/"):
            work_url = work_id_clean.replace("https://openalex.org", "")
        elif work_id_clean.startswith("W"):
            work_url = f"/works/{work_id_clean}"
        else:
            # Assume it's a DOI
            work_url = f"/works/doi:{work_id_clean}"

        response = await client.get(work_url)
        response.raise_for_status()
        source_work = response.json()

        referenced_works = source_work.get("referenced_works", [])
        if not referenced_works:
            logger.debug(f"No referenced works found for {work_id_clean}")
            return OpenAlexCitationResult(
                source_doi=work_id_clean,
                direction="backward",
                total_count=0,
                results=[],
            )

        # Limit referenced works to fetch
        referenced_works = referenced_works[:limit]

        # Extract OpenAlex IDs from URLs (format: https://openalex.org/W123456789)
        openalex_ids = [w.split("/")[-1] for w in referenced_works if w]

        # Fetch full metadata using pipe-delimited filter
        if openalex_ids:
            filter_str = "|".join(openalex_ids)
            params = {
                "filter": f"openalex_id:{filter_str}",
                "per_page": len(openalex_ids),
            }

            response = await client.get("/works", params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for work in data.get("results", []):
                try:
                    parsed = _parse_work(work)
                    results.append(parsed)
                except Exception as e:
                    logger.warning(f"Failed to parse backward citation work: {e}")
                    continue

            logger.debug(
                f"get_backward_citations returned {len(results)} results for {work_id_clean}"
            )

            return OpenAlexCitationResult(
                source_doi=work_id_clean,
                direction="backward",
                total_count=len(referenced_works),
                results=results,
            )
        else:
            return OpenAlexCitationResult(
                source_doi=work_id_clean,
                direction="backward",
                total_count=0,
                results=[],
            )

    except Exception as e:
        logger.error(f"get_backward_citations failed for {work_id_clean}: {e}")
        return OpenAlexCitationResult(
            source_doi=work_id_clean,
            direction="backward",
            total_count=0,
            results=[],
        )


async def get_author_works(
    author_id: str,
    limit: int = 20,
    min_citations: int = 10,
    from_year: Optional[int] = None,
) -> OpenAlexAuthorWorksResult:
    """Get works by a specific author from OpenAlex.

    First fetches author info to get display name, then fetches their works
    filtered by authorship and sorted by citation count.

    Args:
        author_id: OpenAlex author ID (format: A123456789)
        limit: Maximum results (default 20, max 100)
        min_citations: Minimum citation count filter (default 10)
        from_year: Only include works from this year onwards

    Returns:
        OpenAlexAuthorWorksResult with author info and their works
    """
    client = _get_openalex()
    limit = min(max(1, limit), 100)

    try:
        # First, fetch author info
        author_url = f"/authors/{author_id}"
        response = await client.get(author_url)
        response.raise_for_status()
        author_data = response.json()

        author_name = author_data.get("display_name", "Unknown")
        works_count = author_data.get("works_count", 0)

        # Build filter string for works
        filters = [f"authorships.author.id:{author_id}"]
        if min_citations is not None:
            filters.append(f"cited_by_count:>{min_citations}")
        if from_year is not None:
            filters.append(f"publication_year:>{from_year - 1}")

        params = {
            "filter": ",".join(filters),
            "per_page": limit,
            "sort": "cited_by_count:desc",
        }

        response = await client.get("/works", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for work in data.get("results", []):
            try:
                parsed = _parse_work(work)
                results.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse author work: {e}")
                continue

        logger.debug(
            f"get_author_works returned {len(results)} results for {author_id} ({author_name})"
        )

        return OpenAlexAuthorWorksResult(
            author_id=author_id,
            author_name=author_name,
            total_works=works_count,
            results=results,
        )

    except Exception as e:
        logger.error(f"get_author_works failed for {author_id}: {e}")
        return OpenAlexAuthorWorksResult(
            author_id=author_id,
            author_name="Unknown",
            total_works=0,
            results=[],
        )


async def resolve_doi_to_openalex_id(doi: str) -> Optional[str]:
    """Resolve a DOI to its OpenAlex ID.

    Normalizes DOI format and fetches the work to extract OpenAlex ID.

    Args:
        doi: DOI string (with or without https://doi.org/ prefix)

    Returns:
        OpenAlex ID (format: W123456789) or None if not found
    """
    client = _get_openalex()

    # Normalize DOI (remove https://doi.org/ prefix)
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

    try:
        work_url = f"/works/doi:{doi_clean}"
        response = await client.get(work_url)
        response.raise_for_status()
        work_data = response.json()

        openalex_id = work_data.get("id", "")
        if openalex_id:
            # Extract just the ID from the URL (format: https://openalex.org/W123456789)
            openalex_id = openalex_id.split("/")[-1]
            logger.debug(f"Resolved DOI {doi_clean} to OpenAlex ID {openalex_id}")
            return openalex_id
        else:
            logger.warning(f"No OpenAlex ID found for DOI {doi_clean}")
            return None

    except Exception as e:
        logger.error(f"resolve_doi_to_openalex_id failed for {doi_clean}: {e}")
        return None
