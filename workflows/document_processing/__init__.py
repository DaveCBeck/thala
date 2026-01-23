"""Document processing workflow.

Uses Anthropic Claude models for LLM tasks:
- Sonnet: Standard summarization and metadata extraction
- Opus with extended thinking: Complex chapter analysis

Cost optimizations:
- Prompt caching: 90% reduction for repeated tasks
- Batch API (per-document): 50% reduction when use_batch_api=True
"""

from workflows.document_processing.graph import (
    create_document_processing_graph,
    process_document,
    process_documents_batch,
)
from workflows.document_processing.state import (
    ChapterInfo,
    ChapterSummaryState,
    DocumentInput,
    DocumentProcessingState,
    ProcessingResult,
    StoreRecordRef,
    merge_metadata,
)

__all__ = [
    # State types
    "ChapterInfo",
    "ChapterSummaryState",
    "DocumentInput",
    "DocumentProcessingState",
    "ProcessingResult",
    "StoreRecordRef",
    "merge_metadata",
    # Processing functions
    "create_document_processing_graph",
    "process_document",
    "process_documents_batch",
]
