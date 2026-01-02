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

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv

from .llm_utils import ModelTier

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single request to be included in a batch."""
    custom_id: str
    prompt: str
    model: ModelTier
    max_tokens: int = 4096
    system: Optional[str] = None
    thinking_budget: Optional[int] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[dict] = None


@dataclass
class BatchResult:
    """Result from a single batch request."""
    custom_id: str
    success: bool
    content: Optional[str] = None
    thinking: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict] = None


class BatchProcessor:
    """Processor for Anthropic Message Batches API.

    Collects LLM requests and submits them as a batch for 50% cost reduction.
    Results are available when the batch completes (typically within 1 hour).
    """

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

        self.client = Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)
        self.poll_interval = poll_interval
        self.max_wait_hours = max_wait_hours
        self.pending_requests: list[BatchRequest] = []

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
        self.pending_requests.append(BatchRequest(
            custom_id=custom_id,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            system=system,
            thinking_budget=thinking_budget,
            tools=tools,
            tool_choice=tool_choice,
        ))

    def clear_requests(self) -> None:
        """Clear all pending requests."""
        self.pending_requests.clear()

    def _build_batch_requests(self) -> list[dict]:
        """Convert pending requests to API format."""
        batch_requests = []
        for req in self.pending_requests:
            params: dict[str, Any] = {
                "model": req.model.value,
                "max_tokens": req.max_tokens,
                "messages": [{"role": "user", "content": req.prompt}],
            }

            if req.system:
                params["system"] = req.system

            if req.thinking_budget:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": req.thinking_budget,
                }
                # Ensure max_tokens > thinking_budget
                if req.max_tokens <= req.thinking_budget:
                    params["max_tokens"] = req.thinking_budget + 4096

            if req.tools:
                params["tools"] = req.tools
            if req.tool_choice:
                params["tool_choice"] = req.tool_choice

            batch_requests.append({
                "custom_id": req.custom_id,
                "params": params,
            })

        return batch_requests

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

        batch_requests = self._build_batch_requests()
        logger.info(f"Submitting batch with {len(batch_requests)} requests")

        # Create the batch
        batch = self.client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id
        logger.info(f"Created batch {batch_id}, status: {batch.processing_status}")

        # Poll for completion
        max_polls = int(self.max_wait_hours * 3600 / self.poll_interval)
        for poll_count in range(max_polls):
            if batch.processing_status == "ended":
                break

            await asyncio.sleep(self.poll_interval)
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
        results = await self._fetch_results(batch.results_url)

        # Clear pending requests after successful execution
        self.clear_requests()

        return results

    async def _fetch_results(self, results_url: str) -> dict[str, BatchResult]:
        """Fetch and parse batch results from the results URL."""
        results: dict[str, BatchResult] = {}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                results_url,
                headers={
                    "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                    "anthropic-version": "2023-06-01",
                },
            )
            response.raise_for_status()

            # Results are JSONL format (one JSON object per line)
            for line in response.text.strip().split("\n"):
                if not line:
                    continue

                result_data = json.loads(line)
                custom_id = result_data["custom_id"]
                result = result_data["result"]

                if result["type"] == "succeeded":
                    message = result["message"]
                    content = ""
                    thinking = None

                    # Extract content from message
                    for block in message.get("content", []):
                        if block.get("type") == "text":
                            content = block.get("text", "")
                        elif block.get("type") == "thinking":
                            thinking = block.get("thinking", "")
                        elif block.get("type") == "tool_use":
                            # Tool input is already valid JSON - serialize it
                            content = json.dumps(block.get("input", {}))

                    results[custom_id] = BatchResult(
                        custom_id=custom_id,
                        success=True,
                        content=content,
                        thinking=thinking,
                        usage=message.get("usage"),
                    )
                elif result["type"] == "errored":
                    error = result.get("error", {})
                    results[custom_id] = BatchResult(
                        custom_id=custom_id,
                        success=False,
                        error=f"{error.get('type', 'unknown')}: {error.get('message', 'Unknown error')}",
                    )
                elif result["type"] in ("canceled", "expired"):
                    results[custom_id] = BatchResult(
                        custom_id=custom_id,
                        success=False,
                        error=f"Request {result['type']}",
                    )

        return results

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

        batch_requests = self._build_batch_requests()
        batch = self.client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id

        max_polls = int(self.max_wait_hours * 3600 / self.poll_interval)
        polls_per_callback = max(1, callback_interval // self.poll_interval)

        for poll_count in range(max_polls):
            if batch.processing_status == "ended":
                break

            await asyncio.sleep(self.poll_interval)
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

        results = await self._fetch_results(batch.results_url)
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
