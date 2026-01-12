"""
Chapter summarization subgraph using map-reduce pattern.

Uses Opus with extended thinking for complex chapter analysis.
Uses Anthropic Batch API for 50% cost reduction when processing 5+ chapters.
Chunks very long chapters (>600k chars) to avoid token limit errors.
Generates dual summaries (original language + English) for non-English documents.
"""

from .graph import (
    create_chapter_summarization_subgraph,
    chapter_summarization_subgraph,
)
from .nodes import (
    summarize_chapters,
    aggregate_summaries,
)
from .chunking import (
    chunk_large_content,
    MAX_CHAPTER_CHARS,
    CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP_CHARS,
)
from .prompts import (
    CHAPTER_SUMMARIZATION_SYSTEM,
    TRANSLATION_SYSTEM,
)

__all__ = [
    "chapter_summarization_subgraph",
    "create_chapter_summarization_subgraph",
    "summarize_chapters",
    "aggregate_summaries",
    "chunk_large_content",
    "MAX_CHAPTER_CHARS",
    "CHUNK_SIZE_CHARS",
    "CHUNK_OVERLAP_CHARS",
    "CHAPTER_SUMMARIZATION_SYSTEM",
    "TRANSLATION_SYSTEM",
]
