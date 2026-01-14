"""
State schema for document processing workflow.
"""

from datetime import datetime
from operator import add
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict


def merge_metadata(existing: dict, new: dict) -> dict:
    """Merge metadata dicts, new values override existing."""
    return {**existing, **new}


class DocumentInput(TypedDict, total=False):
    """Input specification for document processing."""

    source: str
    title: Optional[str]
    item_type: str
    langs: list[str]
    extra_metadata: dict
    use_batch_api: bool  # Set False for rapid iteration (skips batch API, default True)


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


class StoreRecordRef(TypedDict, total=False):
    """Reference to created store record."""

    id: str
    compression_level: int
    content_preview: str
    language_code: Optional[str]  # ISO 639-1 code for this record


class DocumentProcessingState(TypedDict):
    """Main workflow state."""

    # Input
    input: DocumentInput

    # Source resolution
    source_type: Literal["url", "markdown_text"]
    resolved_path: Optional[str]

    # Zotero tracking
    zotero_key: Optional[str]

    # Store records (parallel writes)
    store_records: Annotated[list[StoreRecordRef], add]

    # Processing
    processing_result: Optional[ProcessingResult]

    # Language detection
    original_language: Optional[str]  # ISO 639-1 code detected from L0 content
    original_language_confidence: Optional[float]  # Detection confidence 0.0-1.0

    # Summary agent output
    short_summary: Optional[str]  # Kept for backward compatibility
    short_summary_original: Optional[str]  # Summary in original language
    short_summary_english: Optional[str]  # English translation (if non-English)

    # Metadata agent output (parallel writes)
    metadata_updates: Annotated[dict, merge_metadata]

    # Chapter detection
    chapters: list[ChapterInfo]
    needs_tenth_summary: bool

    # 10:1 summary
    tenth_summary: Optional[str]  # Kept for backward compatibility
    tenth_summary_original: Optional[str]  # 10:1 summary in original language
    tenth_summary_english: Optional[str]  # English translation (if non-English)
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
