"""DOI detection and resolution subpackage."""

from .detector import detect_doi, extract_doi_from_content
from .resolver import get_oa_url_for_doi, resolve_doi, search_doi_by_title

__all__ = [
    "detect_doi",
    "extract_doi_from_content",
    "resolve_doi",
    "get_oa_url_for_doi",
    "search_doi_by_title",
]
