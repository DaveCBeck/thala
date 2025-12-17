"""
State schema for document processing workflow.
"""

from datetime import datetime
from operator import add
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict
from uuid import UUID


def merge_metadata(existing: dict, new: dict) -> dict:
    """Merge metadata dicts, new values override existing."""
    return {**existing, **new}


class DocumentInput(TypedDict):
    """Input specification for document processing."""
    source: str
    title: Optional[str]
    item_type: str
    quality: Literal["fast", "balanced", "quality"]
    langs: list[str]
    extra_metadata: dict


class ChapterInfo(TypedDict):
    """Chapter metadata for 10:1 summarization."""
    title: str
    start_position: int
    end_position: int
    author: Optional[str]
    word_count: int


class ProcessingResult(TypedDict):
    """Output from Marker or chunker."""
    markdown: str
    chunks: list[dict]
    page_count: int
    word_count: int
    ocr_method: str


class StoreRecordRef(TypedDict):
    """Reference to created store record."""
    id: str
    compression_level: int
    content_preview: str


class DocumentProcessingState(TypedDict):
    """Main workflow state."""
    # Input
    input: DocumentInput

    # Source resolution
    source_type: Literal["url", "local_file", "markdown_text"]
    resolved_path: Optional[str]
    is_already_markdown: bool

    # Zotero tracking
    zotero_key: Optional[str]

    # Store records (parallel writes)
    store_records: Annotated[list[StoreRecordRef], add]

    # Processing
    processing_result: Optional[ProcessingResult]

    # Summary agent output
    short_summary: Optional[str]

    # Metadata agent output (parallel writes)
    metadata_updates: Annotated[dict, merge_metadata]

    # Chapter detection
    chapters: list[ChapterInfo]
    needs_tenth_summary: bool

    # 10:1 summary
    tenth_summary: Optional[str]
    chapter_summaries: Annotated[list[dict], add]

    # Error tracking
    errors: Annotated[list[dict], add]

    # Workflow metadata
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    current_status: str


class ChapterSummaryState(TypedDict):
    """State for chapter summarization subgraph."""
    chapter: ChapterInfo
    chapter_content: str
    target_words: int
    summary: Optional[str]
