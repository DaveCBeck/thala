"""Batch processing infrastructure for Anthropic Message Batches API.

This module provides 50% cost reduction by batching LLM requests for
asynchronous processing. Most batches complete within 1 hour.

Usage:
    processor = BatchProcessor()

    # Add requests
    processor.add_request("summary-1", "Summarize this text...", ModelTier.SONNET)
    processor.add_request("summary-2", "Summarize this other text...", ModelTier.SONNET)

    # Execute batch and get results
    results = await processor.execute_batch()
    print(results["summary-1"])  # Response for first request
"""

from ..llm_utils import ModelTier
from .models import BatchRequest, BatchResult
from .processor import BatchProcessor, get_batch_processor

__all__ = [
    "BatchProcessor",
    "BatchRequest",
    "BatchResult",
    "get_batch_processor",
    "ModelTier",
]
