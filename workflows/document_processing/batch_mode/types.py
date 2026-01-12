"""Batch processing types for document processing."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BatchDocumentRequest:
    """Request for batch document summarization."""
    document_id: str
    content: str
    title: Optional[str] = None
    summary_target_words: int = 100
    include_metadata: bool = True
    include_chapter_summaries: bool = False
    chapters: Optional[list[dict]] = None


@dataclass
class BatchDocumentResult:
    """Result from batch document processing."""
    document_id: str
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    chapter_summaries: Optional[list[dict]] = None
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
