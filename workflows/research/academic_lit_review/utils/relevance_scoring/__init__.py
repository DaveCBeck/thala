"""Relevance scoring utilities for academic literature review workflow.

Contains:
- Relevance scoring prompts and functions
- Batch scoring using Anthropic Batch API (50% cost reduction)
"""

from .scorer import (
    batch_score_relevance,
    score_paper_relevance,
)
from .strategies import (
    chunk_papers,
    format_paper_for_batch,
)
from .types import (
    BATCH_RELEVANCE_SCORING_SYSTEM,
    BATCH_RELEVANCE_SCORING_USER_TEMPLATE,
    RELEVANCE_SCORING_SYSTEM,
    RELEVANCE_SCORING_USER_TEMPLATE,
)

__all__ = [
    "batch_score_relevance",
    "score_paper_relevance",
    "chunk_papers",
    "format_paper_for_batch",
    "BATCH_RELEVANCE_SCORING_SYSTEM",
    "BATCH_RELEVANCE_SCORING_USER_TEMPLATE",
    "RELEVANCE_SCORING_SYSTEM",
    "RELEVANCE_SCORING_USER_TEMPLATE",
]
