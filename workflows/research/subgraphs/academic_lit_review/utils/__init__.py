"""Shared utilities for academic literature review workflow.

Contains:
- OpenAlex result conversion to PaperMetadata
- Relevance scoring prompts and functions
- Deduplication helpers
- Query generation utilities
"""

from .conversion import convert_to_paper_metadata, deduplicate_papers
from .query_generation import generate_search_queries
from .relevance_scoring import batch_score_relevance, score_paper_relevance

__all__ = [
    "convert_to_paper_metadata",
    "deduplicate_papers",
    "score_paper_relevance",
    "batch_score_relevance",
    "generate_search_queries",
]
