"""Document processing workflow.

Uses Anthropic Claude models for LLM tasks:
- Sonnet: Standard summarization and metadata extraction
- Opus with extended thinking: Complex chapter analysis

Supports batch processing via Anthropic's Message Batches API
for 50% cost reduction when processing multiple documents.
"""

from workflows.document_processing.graph import (
    create_document_processing_graph,
    process_document,
    process_documents_batch,
)
from workflows.document_processing.batch_mode import (
    BatchDocumentProcessor,
    BatchDocumentRequest,
    BatchDocumentResult,
    process_documents_with_batch_api,
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
    # Standard processing
    "create_document_processing_graph",
    "process_document",
    "process_documents_batch",
    # Batch API processing (50% cost savings)
    "BatchDocumentProcessor",
    "BatchDocumentRequest",
    "BatchDocumentResult",
    "process_documents_with_batch_api",
]
