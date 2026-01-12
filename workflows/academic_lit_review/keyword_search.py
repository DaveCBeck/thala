"""Keyword search subgraph - compatibility shim.

This module has been refactored into a package structure.
All exports are re-exported from the new location for backward compatibility.
"""

from workflows.academic_lit_review.keyword_search import (
    KeywordSearchState,
    keyword_search_subgraph,
    run_keyword_search,
)

__all__ = [
    "KeywordSearchState",
    "keyword_search_subgraph",
    "run_keyword_search",
]
