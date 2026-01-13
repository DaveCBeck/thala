"""OpenAlex DOI resolution.

Resolves DOIs to metadata and Open Access URLs via OpenAlex API.
"""

import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_TIMEOUT = 30.0


def _get_headers() -> dict[str, str]:
    """Get headers for OpenAlex API requests."""
    headers = {}
    # Use email for polite pool (faster rate limits)
    email = os.getenv("OPENALEX_EMAIL")
    if email:
        headers["User-Agent"] = f"mailto:{email}"
    return headers


async def resolve_doi(doi: str) -> Optional[dict[str, Any]]:
    """Look up DOI in OpenAlex and get metadata.

    Args:
        doi: DOI to resolve (e.g., "10.1234/example")

    Returns:
        OpenAlex work metadata dict if found, None otherwise
    """
    async with httpx.AsyncClient(timeout=OPENALEX_TIMEOUT) as client:
        try:
            response = await client.get(
                f"{OPENALEX_BASE_URL}/works/doi:{doi}",
                headers=_get_headers(),
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.debug(f"DOI not found in OpenAlex: {doi}")
            else:
                logger.warning(
                    f"OpenAlex lookup failed for {doi}: HTTP {response.status_code}"
                )
        except httpx.TimeoutException:
            logger.warning(f"OpenAlex lookup timed out for {doi}")
        except Exception as e:
            logger.warning(f"OpenAlex lookup failed for {doi}: {e}")
    return None


async def get_oa_url_for_doi(doi: str) -> Optional[str]:
    """Get the best open access URL for a DOI via OpenAlex.

    Prefers PDF URLs over landing pages.

    Args:
        doi: DOI to resolve

    Returns:
        Best OA URL if available, None otherwise
    """
    work = await resolve_doi(doi)
    if not work:
        return None

    # Check primary_location first
    primary_location = work.get("primary_location") or {}
    if primary_location:
        # Prefer PDF URL
        pdf_url = primary_location.get("pdf_url")
        if pdf_url:
            return pdf_url
        # Fall back to landing page
        landing_page = primary_location.get("landing_page_url")
        if landing_page:
            return landing_page

    # Check best_oa_location
    best_oa = work.get("best_oa_location") or {}
    if best_oa:
        pdf_url = best_oa.get("pdf_url")
        if pdf_url:
            return pdf_url
        landing_page = best_oa.get("landing_page_url")
        if landing_page:
            return landing_page

    # Check all open_access_locations
    oa_locations = work.get("locations", [])
    for loc in oa_locations:
        if loc.get("is_oa"):
            pdf_url = loc.get("pdf_url")
            if pdf_url:
                return pdf_url

    # Last resort: landing page from any OA location
    for loc in oa_locations:
        if loc.get("is_oa"):
            landing_page = loc.get("landing_page_url")
            if landing_page:
                return landing_page

    return None


async def search_doi_by_title(
    title: str,
    authors: Optional[list[str]] = None,
) -> Optional[str]:
    """Search OpenAlex for a DOI by title (and optionally authors).

    Args:
        title: Article title to search for
        authors: Optional list of author names to improve match accuracy

    Returns:
        DOI string if found with high confidence, None otherwise
    """
    async with httpx.AsyncClient(timeout=OPENALEX_TIMEOUT) as client:
        try:
            # Build search query
            params = {
                "search": title,
                "select": "id,doi,title,authorships",
                "per_page": 5,
            }

            response = await client.get(
                f"{OPENALEX_BASE_URL}/works",
                headers=_get_headers(),
                params=params,
            )

            if response.status_code != 200:
                logger.warning(f"OpenAlex search failed: HTTP {response.status_code}")
                return None

            data = response.json()
            results = data.get("results", [])

            if not results:
                logger.debug(f"No OpenAlex results for title: {title[:50]}...")
                return None

            # Check first result for title match
            best_match = results[0]
            match_title = best_match.get("title", "")
            match_doi = best_match.get("doi")

            if not match_doi:
                return None

            # Extract DOI from URL format (https://doi.org/10.xxxx/yyyy)
            if match_doi.startswith("https://doi.org/"):
                match_doi = match_doi[16:]

            # Simple title similarity check (normalize and compare)
            def normalize(s: str) -> str:
                return "".join(c.lower() for c in s if c.isalnum())

            title_norm = normalize(title)
            match_norm = normalize(match_title)

            # Require reasonable similarity
            if len(title_norm) < 10 or len(match_norm) < 10:
                return None

            # Check if one contains the other or high overlap
            shorter, longer = sorted([title_norm, match_norm], key=len)
            if shorter in longer or len(set(shorter) & set(longer)) / len(shorter) > 0.8:
                logger.info(f"Found DOI via title search: {match_doi}")
                return match_doi

            # If authors provided, check for author match as confirmation
            if authors:
                match_authors = [
                    a.get("author", {}).get("display_name", "").lower()
                    for a in best_match.get("authorships", [])
                ]
                author_match = any(
                    any(auth.lower() in ma for ma in match_authors)
                    for auth in authors
                )
                if author_match:
                    logger.info(f"Found DOI via title+author search: {match_doi}")
                    return match_doi

            logger.debug(f"Title match too weak: '{title[:30]}' vs '{match_title[:30]}'")
            return None

        except httpx.TimeoutException:
            logger.warning(f"OpenAlex search timed out for title: {title[:50]}...")
        except Exception as e:
            logger.warning(f"OpenAlex search failed: {e}")

    return None


async def get_work_metadata(doi: str) -> Optional[dict[str, Any]]:
    """Get structured metadata for a DOI.

    Args:
        doi: DOI to look up

    Returns:
        Dict with title, authors, abstract, year, etc. if found
    """
    work = await resolve_doi(doi)
    if not work:
        return None

    # Extract authors
    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = author.get("display_name")
        if name:
            authors.append(name)

    return {
        "doi": doi,
        "title": work.get("title"),
        "authors": authors,
        "abstract": work.get("abstract"),
        "publication_year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count"),
        "oa_url": await get_oa_url_for_doi(doi),
        "openalex_id": work.get("id"),
    }
