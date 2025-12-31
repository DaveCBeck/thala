"""Query functions for OpenAlex."""

import logging
from typing import Optional

from .client import _get_openalex
from .models import OpenAlexAuthorWorksResult, OpenAlexCitationResult, OpenAlexWork
from .parsing import _parse_work

logger = logging.getLogger(__name__)


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


async def get_work_by_doi(doi: str) -> Optional[OpenAlexWork]:
    """Fetch a work's full metadata by DOI.

    Useful for looking up papers that were discovered through citation analysis
    but weren't in the original search results.

    Args:
        doi: DOI string (with or without https://doi.org/ prefix)

    Returns:
        OpenAlexWork with full metadata, or None if not found
    """
    client = _get_openalex()

    # Normalize DOI (remove https://doi.org/ prefix)
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

    try:
        work_url = f"/works/doi:{doi_clean}"
        response = await client.get(work_url)
        response.raise_for_status()
        work_data = response.json()

        parsed = _parse_work(work_data)
        logger.debug(f"Fetched work by DOI {doi_clean}: {parsed.title[:50]}...")
        return parsed

    except Exception as e:
        logger.debug(f"get_work_by_doi failed for {doi_clean}: {e}")
        return None


async def get_works_by_dois(dois: list[str]) -> list[OpenAlexWork]:
    """Fetch multiple works by their DOIs in a single batch request.

    Uses pipe-delimited DOI filter for efficiency.

    Args:
        dois: List of DOI strings

    Returns:
        List of OpenAlexWork objects for found papers
    """
    if not dois:
        return []

    client = _get_openalex()

    # Normalize DOIs
    dois_clean = [
        d.replace("https://doi.org/", "").replace("http://doi.org/", "")
        for d in dois
    ]

    try:
        # Use pipe-delimited filter for batch lookup
        filter_str = "|".join(f"https://doi.org/{d}" for d in dois_clean)
        params = {
            "filter": f"doi:{filter_str}",
            "per_page": min(len(dois_clean), 50),
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
                logger.warning(f"Failed to parse work in batch: {e}")
                continue

        logger.debug(f"Batch fetched {len(results)}/{len(dois)} works by DOI")
        return results

    except Exception as e:
        logger.error(f"get_works_by_dois failed: {e}")
        return []
