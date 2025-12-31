"""
OpenAlex academic search tools for LangChain.

OpenAlex is a free, open catalog of 240M+ scholarly works.
Provides: openalex_search
"""

from .models import (
    OpenAlexAuthor,
    OpenAlexAuthorWorksResult,
    OpenAlexCitationResult,
    OpenAlexSearchOutput,
    OpenAlexWork,
)
from .queries import (
    get_author_works,
    get_backward_citations,
    get_forward_citations,
    get_work_by_doi,
    get_works_by_dois,
    resolve_doi_to_openalex_id,
)
from .tools import openalex_search

__all__ = [
    "OpenAlexAuthor",
    "OpenAlexWork",
    "OpenAlexSearchOutput",
    "OpenAlexCitationResult",
    "OpenAlexAuthorWorksResult",
    "openalex_search",
    "get_forward_citations",
    "get_backward_citations",
    "get_author_works",
    "resolve_doi_to_openalex_id",
    "get_work_by_doi",
    "get_works_by_dois",
]
