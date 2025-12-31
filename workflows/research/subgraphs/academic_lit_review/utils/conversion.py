"""Conversion utilities for academic literature review workflow.

Contains:
- OpenAlex result conversion to PaperMetadata
- Deduplication helpers
"""

import logging
from datetime import datetime
from typing import Optional

from langchain_tools.openalex import OpenAlexWork
from workflows.research.subgraphs.academic_lit_review.state import (
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
        oa_status=work_dict.get("oa_status"),
        referenced_works=work_dict.get("referenced_works", []),
        citing_works_count=work_dict.get("cited_by_count", 0),
        retrieved_at=datetime.utcnow(),
        discovery_stage=discovery_stage,
        discovery_method=discovery_method,
    )


def deduplicate_papers(
    papers: list[PaperMetadata],
    existing_dois: set[str] | None = None,
) -> list[PaperMetadata]:
    """Deduplicate papers by DOI, keeping the first occurrence.

    Args:
        papers: List of PaperMetadata to deduplicate
        existing_dois: Optional set of DOIs already in corpus to exclude

    Returns:
        Deduplicated list of papers
    """
    existing_dois = existing_dois or set()
    seen_dois = set()
    unique_papers = []

    for paper in papers:
        doi = paper.get("doi")
        if doi and doi not in seen_dois and doi not in existing_dois:
            seen_dois.add(doi)
            unique_papers.append(paper)

    logger.debug(
        f"Deduplicated {len(papers)} papers to {len(unique_papers)} "
        f"(excluded {len(papers) - len(unique_papers)} duplicates)"
    )
    return unique_papers
