"""Batch processing infrastructure for Anthropic Message Batches API.

This module provides 50% cost reduction by batching LLM requests for
asynchronous processing. Most batches complete within 1 hour.

LangSmith Integration:
- Clients wrapped with langsmith.wrappers.wrap_anthropic for tracing
- Batch execution methods decorated with @traceable for visibility
- Token usage aggregated and attached to run metadata
"""

import asyncio
import logging
import os
from typing import Optional

from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv
from langsmith import get_current_run_tree, traceable
from langsmith.wrappers import wrap_anthropic

from ..llm_utils import ModelTier
from .models import BatchRequest, BatchResult
from .request_builder import RequestBuilder
from .result_parser import ResultParser

load_dotenv()
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processor for Anthropic Message Batches API.

    Collects LLM requests and submits them as a batch for 50% cost reduction.
    Results are available when the batch completes (typically within 1 hour).
    """

    # Beta header for 1M context window (Sonnet 4/4.5 only, Tier 4+)
    CONTEXT_1M_BETA = "context-1m-2025-08-07"

    def __init__(self, poll_interval: int = 60, max_wait_hours: float = 24):
        """
        Initialize batch processor.

        Args:
            poll_interval: Seconds between status checks (default: 60)
            max_wait_hours: Maximum hours to wait for batch completion (default: 24)
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        # Wrap clients with LangSmith for tracing batch API calls
        self.client = wrap_anthropic(Anthropic(api_key=api_key))
        self.async_client = wrap_anthropic(AsyncAnthropic(api_key=api_key))
        self.poll_interval = poll_interval
        self.max_wait_hours = max_wait_hours
        self.pending_requests: list[BatchRequest] = []
        self._request_builder = RequestBuilder()
        self._result_parser = ResultParser(self._request_builder.get_original_id)
        # Track if any request needs 1M context
        self._needs_1m_context: bool = False

    def add_request(
        self,
        custom_id: str,
        prompt: str,
        model: ModelTier = ModelTier.SONNET,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[dict] = None,
    ) -> None:
        """
        Add a request to the pending batch.

        Args:
            custom_id: Unique identifier for this request (used to retrieve results)
            prompt: The prompt/message to send to Claude
            model: Model tier to use (default: SONNET)
            max_tokens: Maximum output tokens
            system: Optional system prompt
            thinking_budget: Optional token budget for extended thinking
            tools: Optional list of tool definitions for structured output
            tool_choice: Optional tool choice config (e.g., {"type": "tool", "name": "..."})
        """
        self.pending_requests.append(
            BatchRequest(
                custom_id=custom_id,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                system=system,
                thinking_budget=thinking_budget,
                tools=tools,
                tool_choice=tool_choice,
            )
        )
        # Track if 1M context is needed for this batch
        if model == ModelTier.SONNET_1M:
            self._needs_1m_context = True

    def clear_requests(self) -> None:
        """Clear all pending requests."""
        self.pending_requests.clear()
        self._request_builder.clear()
        self._needs_1m_context = False

    @traceable(name="anthropic_batch_execute", run_type="llm")
    async def execute_batch(self) -> dict[str, BatchResult]:
        """
        Submit batch and wait for results.

        Returns:
            Dictionary mapping custom_id to BatchResult

        Raises:
            RuntimeError: If batch processing fails or times out
        """
        if not self.pending_requests:
            return {}

        batch_requests = self._request_builder.build_batch_requests(
            self.pending_requests
        )
        logger.info(f"Submitting batch with {len(batch_requests)} requests")

        # Create the batch - use beta API if 1M context is needed
        if self._needs_1m_context:
            logger.debug("Using 1M context window beta for this batch")
            batch = self.client.beta.messages.batches.create(
                requests=batch_requests,
                betas=[self.CONTEXT_1M_BETA],
            )
        else:
            batch = self.client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id
        logger.info(f"Created batch {batch_id}, status: {batch.processing_status}")

        # Poll for completion
        max_polls = int(self.max_wait_hours * 3600 / self.poll_interval)
        for poll_count in range(max_polls):
            if batch.processing_status == "ended":
                break

            await asyncio.sleep(self.poll_interval)
            # Use same API for retrieval as for creation
            if self._needs_1m_context:
                batch = self.client.beta.messages.batches.retrieve(batch_id)
            else:
                batch = self.client.messages.batches.retrieve(batch_id)
            logger.debug(
                f"Batch {batch_id} status: {batch.processing_status}, "
                f"counts: {batch.request_counts}"
            )

            if batch.processing_status == "ended":
                break
        else:
            raise RuntimeError(
                f"Batch {batch_id} did not complete within {self.max_wait_hours} hours"
            )

        # Fetch and parse results
        results = await self._result_parser.fetch_results(batch.results_url)

        # Aggregate token usage and attach to LangSmith run
        self._attach_usage_metadata(results, batch_id, len(batch_requests))

        # Clear pending requests after successful execution
        self.clear_requests()

        return results

    def _attach_usage_metadata(
        self,
        results: dict[str, BatchResult],
        batch_id: str,
        request_count: int,
    ) -> None:
        """Attach aggregated token usage to current LangSmith run."""
        total_input = sum(
            r.usage.get("input_tokens", 0) for r in results.values() if r.usage
        )
        total_output = sum(
            r.usage.get("output_tokens", 0) for r in results.values() if r.usage
        )
        total_cache_read = sum(
            r.usage.get("cache_read_input_tokens", 0)
            for r in results.values()
            if r.usage
        )
        total_cache_creation = sum(
            r.usage.get("cache_creation_input_tokens", 0)
            for r in results.values()
            if r.usage
        )

        run = get_current_run_tree()
        if run:
            run.add_metadata(
                {
                    "batch_id": batch_id,
                    "request_count": request_count,
                    "successful_count": sum(1 for r in results.values() if r.success),
                    "failed_count": sum(1 for r in results.values() if not r.success),
                    "usage_metadata": {
                        "input_tokens": total_input,
                        "output_tokens": total_output,
                        "total_tokens": total_input + total_output,
                        "cache_read_input_tokens": total_cache_read,
                        "cache_creation_input_tokens": total_cache_creation,
                    },
                }
            )

    @traceable(name="anthropic_batch_execute_with_callback", run_type="llm")
    async def execute_batch_with_callback(
        self,
        callback: callable,
        callback_interval: int = 300,
    ) -> dict[str, BatchResult]:
        """
        Submit batch and wait for results with periodic callbacks.

        Args:
            callback: Async function called periodically with (batch_id, status, counts)
            callback_interval: Seconds between callbacks (default: 300 = 5 min)

        Returns:
            Dictionary mapping custom_id to BatchResult
        """
        if not self.pending_requests:
            return {}

        batch_requests = self._request_builder.build_batch_requests(
            self.pending_requests
        )

        # Create the batch - use beta API if 1M context is needed
        if self._needs_1m_context:
            logger.debug("Using 1M context window beta for this batch")
            batch = self.client.beta.messages.batches.create(
                requests=batch_requests,
                betas=[self.CONTEXT_1M_BETA],
            )
        else:
            batch = self.client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id

        max_polls = int(self.max_wait_hours * 3600 / self.poll_interval)
        polls_per_callback = max(1, callback_interval // self.poll_interval)

        for poll_count in range(max_polls):
            if batch.processing_status == "ended":
                break

            await asyncio.sleep(self.poll_interval)
            # Use same API for retrieval as for creation
            if self._needs_1m_context:
                batch = self.client.beta.messages.batches.retrieve(batch_id)
            else:
                batch = self.client.messages.batches.retrieve(batch_id)

            # Call callback periodically
            if poll_count % polls_per_callback == 0:
                await callback(batch_id, batch.processing_status, batch.request_counts)

            if batch.processing_status == "ended":
                break
        else:
            raise RuntimeError(
                f"Batch {batch_id} did not complete within {self.max_wait_hours} hours"
            )

        results = await self._result_parser.fetch_results(batch.results_url)

        # Aggregate token usage and attach to LangSmith run
        self._attach_usage_metadata(results, batch_id, len(batch_requests))

        self.clear_requests()
        return results


# Global batch processor instance for convenience
_default_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """Get the default batch processor instance."""
    global _default_processor
    if _default_processor is None:
        _default_processor = BatchProcessor()
    return _default_processor
