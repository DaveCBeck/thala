"""Paper search tools for LLM use in supervision loops.

Provides two-stage retrieval:
1. search_papers: Hybrid (semantic + keyword) search returning compact metadata
2. get_paper_content: Fetch detailed L2 content for specific papers
"""

from .searcher import create_paper_tools, MINIMUM_RELEVANCE_THRESHOLD
from .types import PaperSearchResult, SearchPapersOutput, PaperContentOutput
from .sources import semantic_search, keyword_search, merge_search_results

__all__ = [
    "create_paper_tools",
    "MINIMUM_RELEVANCE_THRESHOLD",
    "PaperSearchResult",
    "SearchPapersOutput",
    "PaperContentOutput",
    "semantic_search",
    "keyword_search",
    "merge_search_results",
]
