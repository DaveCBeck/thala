"""
Batch-mode document processing using Anthropic Message Batches API.

This module provides 50% cost reduction by batching all LLM calls
together and submitting them to Anthropic's Message Batches API.

Use this when:
- Processing multiple documents that don't need immediate results
- Cost savings are more important than latency
- You're doing bulk processing or large-scale analysis

Note: Batch processing is asynchronous. Results may take up to 24 hours
(typically completes within 1 hour).
"""

from .processor import BatchDocumentProcessor, process_documents_with_batch_api
from .types import BatchDocumentRequest, BatchDocumentResult

__all__ = [
    "BatchDocumentProcessor",
    "BatchDocumentRequest",
    "BatchDocumentResult",
    "process_documents_with_batch_api",
]
