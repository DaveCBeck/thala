"""Conversion utilities for academic literature review workflow.

Contains:
- OpenAlex result conversion to PaperMetadata
- Deduplication helpers (DOI-based and title+author fuzzy matching)
"""

import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional

from langchain_tools.openalex import OpenAlexWork
from workflows.research.academic_lit_review.state import (
    PaperAuthor,
    PaperMetadata,
)

logger = logging.getLogger(__name__)


def convert_to_paper_metadata(
    work: OpenAlexWork | dict,
    discovery_stage: int = 0,
    discovery_method: str = "keyword",
) -> Optional[PaperMetadata]:
    """Convert an OpenAlex work to PaperMetadata.

    Args:
        work: OpenAlex work object or dict
        discovery_stage: Which diffusion stage discovered this paper
        discovery_method: How this paper was discovered

    Returns:
        PaperMetadata or None if missing required fields (DOI)
    """
    # Handle both Pydantic model and dict
    if hasattr(work, "model_dump"):
        work_dict = work.model_dump()
    elif hasattr(work, "dict"):
        work_dict = work.dict()
    else:
        work_dict = dict(work)

    doi = work_dict.get("doi")
    if not doi:
        logger.debug(f"Skipping work without DOI: {work_dict.get('title', 'Unknown')}")
        return None

    # Normalize DOI (remove https://doi.org/ prefix)
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

    # Parse authors
    authors = []
    for author_data in work_dict.get("authors", []):
        if isinstance(author_data, dict):
            authors.append(
                PaperAuthor(
                    name=author_data.get("name", "Unknown"),
                    author_id=author_data.get("author_id"),
                    institution=author_data.get("institution"),
                    orcid=author_data.get("orcid"),
                )
            )

    # Extract year from publication_date
    pub_date = work_dict.get("publication_date", "")
    year = 0
    if pub_date and len(pub_date) >= 4:
        try:
            year = int(pub_date[:4])
        except ValueError:
            pass

    # Extract OpenAlex ID from URL if present
    openalex_id = ""
    work_id = work_dict.get("id", "") or work_dict.get("openalex_id", "")
    if work_id:
        openalex_id = work_id.split("/")[-1] if "/" in work_id else work_id

    return PaperMetadata(
        doi=doi_clean,
        title=work_dict.get("title", "Untitled"),
        authors=authors,
        publication_date=pub_date,
        year=year,
        venue=work_dict.get("source_name"),
        cited_by_count=work_dict.get("cited_by_count", 0),
        abstract=work_dict.get("abstract"),
        openalex_id=openalex_id,
        primary_topic=work_dict.get("primary_topic"),
        is_oa=work_dict.get("is_oa", False),
        oa_url=work_dict.get("oa_url"),
        oa_urls=work_dict.get("oa_urls", []),
        pmcid=work_dict.get("pmcid"),
        oa_status=work_dict.get("oa_status"),
        referenced_works=work_dict.get("referenced_works", []),
        citing_works_count=work_dict.get("cited_by_count", 0),
        retrieved_at=datetime.now(timezone.utc),
        discovery_stage=discovery_stage,
        discovery_method=discovery_method,
    )


def _normalize_title(title: str) -> str:
    """Normalize a title for fuzzy comparison.

    Lowercases, strips accents, removes non-alphanumeric characters,
    and collapses whitespace.
    """
    t = title.lower()
    # Strip accents
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    # Keep only alphanumeric and spaces
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _extract_author_lastnames(paper: PaperMetadata, limit: int = 3) -> list[str]:
    """Extract normalized last names from the first N authors."""
    names = []
    for author in paper.get("authors", [])[:limit]:
        full = author.get("name", "")
        if not full:
            continue
        # Take last whitespace-separated token as surname
        surname = full.strip().split()[-1].lower()
        surname = unicodedata.normalize("NFKD", surname)
        surname = "".join(c for c in surname if not unicodedata.combining(c))
        names.append(surname)
    return names


def _is_title_author_duplicate(a: PaperMetadata, b: PaperMetadata) -> bool:
    """Detect whether two papers are the same work (e.g. preprint + published).

    Matches when normalized titles are identical and the first 2+ authors share
    surnames.  This catches the common case where OpenAlex indexes a preprint
    and its published version as separate works with different DOIs.
    """
    title_a = _normalize_title(a.get("title", ""))
    title_b = _normalize_title(b.get("title", ""))
    if not title_a or not title_b:
        return False
    if title_a != title_b:
        return False

    authors_a = _extract_author_lastnames(a)
    authors_b = _extract_author_lastnames(b)
    if not authors_a or not authors_b:
        # Can't confirm — titles match but no author data; be conservative
        return False

    # Require at least 2 shared surnames (or all if fewer than 2 authors)
    shared = set(authors_a) & set(authors_b)
    required = min(2, len(authors_a), len(authors_b))
    return len(shared) >= required


def _pick_preferred_version(a: PaperMetadata, b: PaperMetadata) -> PaperMetadata:
    """Given two versions of the same paper, keep the better one.

    Prefers: higher citation count > later publication date > first encountered.
    """
    if a.get("cited_by_count", 0) != b.get("cited_by_count", 0):
        return a if a.get("cited_by_count", 0) >= b.get("cited_by_count", 0) else b
    if (a.get("year") or 0) != (b.get("year") or 0):
        return a if (a.get("year") or 0) >= (b.get("year") or 0) else b
    return a  # default to first encountered


def deduplicate_papers(
    papers: list[PaperMetadata],
    existing_dois: set[str] | None = None,
    existing_papers: list[PaperMetadata] | None = None,
) -> list[PaperMetadata]:
    """Deduplicate papers by DOI and by title+author similarity.

    Two-pass deduplication:
    1. Exact DOI match (fast, handles most cases)
    2. Title+author match (catches preprint/published duplicates with different DOIs)

    When a title+author duplicate is found, the version with more citations is kept.

    Args:
        papers: List of PaperMetadata to deduplicate
        existing_dois: Optional set of DOIs already in corpus to exclude
        existing_papers: Optional list of papers already in corpus, used for
            title+author dedup against the existing corpus

    Returns:
        Deduplicated list of papers
    """
    existing_dois = existing_dois or set()
    existing_papers = existing_papers or []

    # Pass 1: DOI dedup
    seen_dois: set[str] = set()
    doi_unique: list[PaperMetadata] = []
    for paper in papers:
        doi = paper.get("doi")
        if doi and doi not in seen_dois and doi not in existing_dois:
            seen_dois.add(doi)
            doi_unique.append(paper)

    # Pass 2: title+author dedup within the batch
    title_unique: list[PaperMetadata] = []
    for paper in doi_unique:
        is_dup = False
        for i, kept in enumerate(title_unique):
            if _is_title_author_duplicate(paper, kept):
                preferred = _pick_preferred_version(kept, paper)
                dropped = paper if preferred is kept else kept
                logger.info(
                    "Dropping duplicate: '%s' (DOI %s) — same work as '%s' (DOI %s)",
                    dropped.get("title", "?"),
                    dropped.get("doi", "?"),
                    preferred.get("title", "?"),
                    preferred.get("doi", "?"),
                )
                title_unique[i] = preferred
                is_dup = True
                break
        if not is_dup:
            title_unique.append(paper)

    # Pass 3: title+author dedup against existing corpus
    if existing_papers:
        final: list[PaperMetadata] = []
        for paper in title_unique:
            dup_found = False
            for existing in existing_papers:
                if _is_title_author_duplicate(paper, existing):
                    logger.info(
                        "Dropping corpus duplicate: '%s' (DOI %s) — "
                        "already in corpus as '%s' (DOI %s)",
                        paper.get("title", "?"),
                        paper.get("doi", "?"),
                        existing.get("title", "?"),
                        existing.get("doi", "?"),
                    )
                    dup_found = True
                    break
            if not dup_found:
                final.append(paper)
        title_unique = final

    removed = len(papers) - len(title_unique)
    if removed:
        logger.debug(
            "Deduplicated %d papers to %d (removed %d: DOI + title/author)",
            len(papers), len(title_unique), removed,
        )
    return title_unique
