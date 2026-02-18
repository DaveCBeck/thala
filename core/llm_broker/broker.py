"""Central LLM Broker for routing requests through batch or sync APIs.

This module provides the main LLMBroker class that:
- Routes requests based on user mode and call-site policy
- Manages async futures for awaitable request resolution
- Submits batches to Anthropic Batch API
- Monitors batch status with retry/fallback logic
- Coordinates cross-process access via persistent queue
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic, RateLimitError
from langsmith import traceable
from langsmith.wrappers import wrap_anthropic

from core.task_queue.shutdown import get_shutdown_coordinator
from core.types import ModelTier

from .config import BrokerConfig, get_broker_config
from .exceptions import (
    BatchSubmissionError,
    QueueOverflowError,
)
from .message_params import (
    CONTEXT_1M_BETA,
    build_message_params,
    parse_response_content_with_blocks,
    sanitize_custom_id,
)
from .metrics import BrokerMetrics
from .persistence import BrokerPersistence
from .result_fetcher import fetch_batch_results
from .routing import should_batch
from .schemas import (
    BatchPolicy,
    LLMRequest,
    LLMResponse,
    RequestState,
    UserMode,
)

logger = logging.getLogger(__name__)

# Context variable for batch group isolation across concurrent async contexts
# This ensures parallel LangGraph nodes don't interfere with each other's batch groups
_current_batch_group: ContextVar["BatchGroup | None"] = ContextVar(
    "llm_broker_batch_group", default=None
)

# Backward-compatible alias for test imports
_sanitize_custom_id = sanitize_custom_id


@dataclass
class BatchGroup:
    """Context for grouping requests in a batch.

    Tracks requests added during the batch_group context and triggers
    batch submission on context exit.
    """

    broker: "LLMBroker"
    request_ids: list[str] = field(default_factory=list)
    mode: UserMode = UserMode.BALANCED

    def add_request(self, request_id: str) -> None:
        """Track a request added to this group."""
        self.request_ids.append(request_id)


class LLMBroker:
    """Central broker for routing LLM requests.

    Routes requests through batch or synchronous APIs based on user mode
    and call-site policy. Provides async futures for awaitable resolution
    and handles batch submission, monitoring, and retry logic.

    Usage:
        broker = LLMBroker()
        await broker.start()

        async with broker.batch_group() as group:
            future1 = await broker.request(prompt1, policy=BatchPolicy.PREFER_SPEED)
            future2 = await broker.request(prompt2, policy=BatchPolicy.PREFER_SPEED)
            results = await asyncio.gather(future1, future2)

        await broker.stop()
    """

    def __init__(
        self,
        config: BrokerConfig | None = None,
        mode: UserMode | None = None,
    ) -> None:
        """Initialize the broker.

        Args:
            config: Broker configuration (uses global config if not provided)
            mode: Override default user mode
        """
        self._config = config or get_broker_config()
        self._mode = mode or self._config.default_mode
        self._persistence = BrokerPersistence(self._config.queue_dir)
        self._metrics = BrokerMetrics() if self._config.enable_metrics else None

        # Anthropic client (wrapped for LangSmith tracing)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self._async_client: AsyncAnthropic | None = wrap_anthropic(AsyncAnthropic(api_key=api_key))
        else:
            self._async_client = None
            logger.warning("ANTHROPIC_API_KEY not set - broker will only work for DeepSeek")

        # Concurrency controls
        self._submission_lock = asyncio.Lock()
        self._sync_semaphore = asyncio.Semaphore(self._config.max_concurrent_sync)

        # Future tracking for request resolution
        self._pending_futures: dict[str, asyncio.Future[LLMResponse]] = {}
        self._id_mapping: dict[str, str] = {}  # sanitized -> original request_id

        # Background task
        self._batch_monitor_task: asyncio.Task | None = None
        self._started = False

        # Track fire-and-forget sync tasks to ensure graceful shutdown
        self._sync_tasks: set[asyncio.Task[None]] = set()

    @property
    def mode(self) -> UserMode:
        """Current user mode."""
        return self._mode

    @mode.setter
    def mode(self, value: UserMode) -> None:
        """Set user mode."""
        self._mode = value

    @property
    def metrics(self) -> BrokerMetrics | None:
        """Broker metrics (None if disabled)."""
        return self._metrics

    # Lifecycle

    async def start(self) -> None:
        """Start the broker and background tasks.

        Must be called before making requests. Initializes persistence
        and starts the batch monitor loop.
        """
        if self._started:
            return

        await self._persistence.initialize()

        # Start batch monitor background task
        self._batch_monitor_task = asyncio.create_task(
            self._batch_monitor_loop(),
            name="broker-batch-monitor",
        )
        self._started = True
        logger.info(f"LLM Broker started in {self._mode.value} mode")

    async def stop(self) -> None:
        """Stop the broker and cleanup.

        Cancels background tasks and flushes any remaining queued requests.
        Waits for all in-flight sync tasks to complete gracefully.
        """
        if not self._started:
            return

        # Cancel monitor task
        if self._batch_monitor_task:
            self._batch_monitor_task.cancel()
            try:
                await self._batch_monitor_task
            except asyncio.CancelledError:
                pass
            self._batch_monitor_task = None

        # Wait for all in-flight sync tasks to complete
        if self._sync_tasks:
            logger.info(f"Waiting for {len(self._sync_tasks)} in-flight sync tasks to complete")
            await asyncio.gather(*self._sync_tasks, return_exceptions=True)
            self._sync_tasks.clear()

        # Log final metrics
        if self._metrics:
            self._metrics.log_summary()

        self._started = False
        logger.info("LLM Broker stopped")

    async def __aenter__(self) -> "LLMBroker":
        """Async context manager entry - starts the broker."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - stops the broker."""
        await self.stop()

    # Request submission

    def _spawn_sync_task(self, request: LLMRequest) -> asyncio.Task[None]:
        """Spawn a tracked sync execution task.

        Creates an asyncio task for sync execution and tracks it in _sync_tasks
        to ensure graceful shutdown. Uses a done callback to automatically
        remove completed tasks and log any exceptions.

        Args:
            request: The request to execute synchronously

        Returns:
            The created task
        """
        task = asyncio.create_task(
            self._execute_sync(request),
            name=f"sync-{request.request_id[:8]}",
        )
        self._sync_tasks.add(task)

        def _on_task_done(t: asyncio.Task[None]) -> None:
            self._sync_tasks.discard(t)
            # Log exceptions that would otherwise be orphaned
            if not t.cancelled() and t.exception() is not None:
                logger.error(
                    f"Sync task {t.get_name()} failed with exception: {t.exception()}"
                )

        task.add_done_callback(_on_task_done)
        return task

    @asynccontextmanager
    async def batch_group(
        self,
        mode: UserMode | None = None,
    ) -> AsyncIterator[BatchGroup]:
        """Context manager for grouping requests into a batch.

        Requests made within this context are queued together and
        submitted as a batch when the context exits.

        Uses contextvars for isolation, ensuring parallel LangGraph nodes
        or concurrent async contexts don't interfere with each other's
        batch groups.

        Args:
            mode: Override mode for this batch group

        Yields:
            BatchGroup context for tracking requests
        """
        group = BatchGroup(broker=self, mode=mode or self._mode)
        token = _current_batch_group.set(group)

        try:
            yield group
        finally:
            _current_batch_group.reset(token)

            # Flush queued requests from this group
            if group.request_ids:
                await self._flush_batch_group(group)

    async def request(
        self,
        prompt: str,
        model: ModelTier = ModelTier.SONNET,
        policy: BatchPolicy = BatchPolicy.PREFER_SPEED,
        max_tokens: int = 4096,
        system: str | None = None,
        effort: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> asyncio.Future[LLMResponse]:
        """Submit a request to the broker.

        Routes the request through batch or sync API based on mode and policy.
        Returns a Future that resolves when the response is available.

        Args:
            prompt: User message/prompt
            model: Model tier to use
            policy: Batch policy for this request
            max_tokens: Maximum output tokens
            system: Optional system prompt
            effort: Optional adaptive thinking effort level
            tools: Optional tool definitions
            tool_choice: Optional tool choice config
            metadata: Optional request metadata
            messages: Optional pre-built messages list (overrides prompt)

        Returns:
            Future that resolves to LLMResponse

        Raises:
            QueueOverflowError: If queue full and overflow_behavior is "reject"
        """
        if not self._started:
            await self.start()

        # Create request
        request = LLMRequest.create(
            prompt=prompt,
            model=model.value,
            policy=policy,
            max_tokens=max_tokens,
            system=system,
            effort=effort,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata or {},
            messages=messages,
        )

        # Create future for this request
        loop = asyncio.get_running_loop()
        future: asyncio.Future[LLMResponse] = loop.create_future()
        self._pending_futures[request.request_id] = future

        # Determine routing
        if should_batch(self._mode, policy, model, effort):
            await self._queue_for_batch(request)
            if self._metrics:
                self._metrics.record_request(batched=True)
        else:
            # Execute synchronously with tracked task
            self._spawn_sync_task(request)
            if self._metrics:
                self._metrics.record_request(batched=False)

        return future

    def _should_batch(
        self,
        policy: BatchPolicy,
        model: ModelTier,
        effort: str | None,
    ) -> bool:
        """Determine if a request should be batched.

        Delegates to routing.should_batch() but preserves the instance method
        interface for test compatibility.
        """
        return should_batch(self._mode, policy, model, effort)

    # Queue management

    async def _queue_for_batch(self, request: LLMRequest) -> None:
        """Queue a request for batch processing.

        Args:
            request: The request to queue

        Raises:
            QueueOverflowError: If queue full and overflow_behavior is "reject"
        """
        # Check queue size
        queue_size = await self._persistence.get_queue_size()
        if queue_size >= self._config.max_queue_size:
            if self._metrics:
                self._metrics.record_queue_overflow()

            if self._config.overflow_behavior == "reject":
                raise QueueOverflowError(queue_size, self._config.max_queue_size)
            else:
                # Fall back to sync with tracked task
                logger.warning(
                    f"Queue overflow ({queue_size}/{self._config.max_queue_size}), "
                    f"falling back to sync for request {request.request_id}"
                )
                self._spawn_sync_task(request)
                return

        # Add to persistent queue
        await self._persistence.add_request(request)

        # Track in current batch group if active (uses contextvars for isolation)
        current_group = _current_batch_group.get()
        if current_group:
            current_group.add_request(request.request_id)

        # Check if threshold reached
        await self._check_batch_triggers()

    async def _check_batch_triggers(self) -> None:
        """Check if batch should be submitted based on queue size."""
        async with self._submission_lock:
            queue_size = await self._persistence.get_queue_size()
            if queue_size >= self._config.batch_threshold:
                await self._submit_batch()

    async def _flush_batch_group(self, group: BatchGroup) -> None:
        """Submit batch for requests in a batch group.

        Args:
            group: The batch group to flush
        """
        if not group.request_ids:
            return

        async with self._submission_lock:
            # Get queued requests from this group
            queued = await self._persistence.get_queued_requests()
            group_requests = [r for r in queued if r.request_id in group.request_ids]

            if group_requests:
                await self._submit_batch(group_requests)

    async def flush(self) -> None:
        """Manually flush all queued requests as a batch."""
        async with self._submission_lock:
            await self._submit_batch()

    # Batch submission

    @traceable(name="llm_broker_batch_submit", run_type="chain")
    async def _submit_batch(
        self,
        requests: list[LLMRequest] | None = None,
    ) -> None:
        """Submit a batch of requests to Anthropic Batch API.

        Args:
            requests: Specific requests to submit (or all queued if None)
        """
        if not self._async_client:
            logger.error("Cannot submit batch: Anthropic client not configured")
            return

        # Get requests to submit
        # Note: get_queued_requests() handles its own locking internally,
        # so no outer lock needed here (nesting would deadlock on fcntl flock)
        if requests is None:
            requests = await self._persistence.get_queued_requests()

        if not requests:
            return

        # Build batch requests
        batch_requests = []
        needs_1m_context = False

        for req in requests:
            sanitized_id = sanitize_custom_id(req.request_id)
            self._id_mapping[sanitized_id] = req.request_id

            params = build_message_params(req)

            batch_requests.append(
                {
                    "custom_id": sanitized_id,
                    "params": params,
                }
            )

            # Check if 1M context needed
            if req.model == ModelTier.SONNET_1M.value:
                needs_1m_context = True

        # Submit batch
        try:
            logger.info(f"Submitting batch with {len(batch_requests)} requests")

            if needs_1m_context:
                batch = await self._async_client.beta.messages.batches.create(
                    requests=batch_requests,
                    betas=[CONTEXT_1M_BETA],
                )
            else:
                batch = await self._async_client.messages.batches.create(
                    requests=batch_requests,
                )

            batch_id = batch.id
            logger.info(f"Created batch {batch_id}, status: {batch.processing_status}")

            # Mark requests as submitted
            request_ids = [r.request_id for r in requests]
            await self._persistence.mark_requests_submitted(request_ids, batch_id)

            if self._metrics:
                self._metrics.record_batch_submitted(len(requests))

        except Exception as e:
            logger.error(f"Failed to submit batch: {e}")
            # Resolve futures with errors
            for req in requests:
                self._resolve_future(
                    req.request_id,
                    LLMResponse(
                        request_id=req.request_id,
                        content=None,
                        success=False,
                        error=f"Batch submission failed: {e}",
                    ),
                )
            raise BatchSubmissionError(len(requests), str(e))

    # Synchronous execution

    @traceable(name="llm_broker_sync", run_type="llm")
    async def _execute_sync(self, request: LLMRequest) -> None:
        """Execute a request via synchronous (non-batch) API.

        Args:
            request: The request to execute
        """
        if not self._async_client:
            self._resolve_future(
                request.request_id,
                LLMResponse(
                    request_id=request.request_id,
                    content=None,
                    success=False,
                    error="Anthropic client not configured",
                ),
            )
            return

        async with self._sync_semaphore:
            try:
                kwargs = build_message_params(request)

                # Use streaming to avoid 10-minute timeout on long requests
                if request.model == ModelTier.SONNET_1M.value:
                    async with self._async_client.beta.messages.stream(
                        **kwargs,
                        betas=[CONTEXT_1M_BETA],
                    ) as stream:
                        response = await stream.get_final_message()
                else:
                    async with self._async_client.messages.stream(**kwargs) as stream:
                        response = await stream.get_final_message()

                content, thinking, content_blocks = parse_response_content_with_blocks(response)

                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                if getattr(response.usage, "cache_creation_input_tokens", None):
                    usage["cache_creation_input_tokens"] = response.usage.cache_creation_input_tokens
                if getattr(response.usage, "cache_read_input_tokens", None):
                    usage["cache_read_input_tokens"] = response.usage.cache_read_input_tokens

                self._resolve_future(
                    request.request_id,
                    LLMResponse(
                        request_id=request.request_id,
                        content=content,
                        success=True,
                        usage=usage,
                        model=response.model,
                        stop_reason=response.stop_reason,
                        thinking=thinking,
                        content_blocks=content_blocks,
                    ),
                )

            except RateLimitError as e:
                logger.warning(f"Rate limited for {request.request_id}, will retry in 60s: {e}")
                await asyncio.sleep(60)
                self._spawn_sync_task(request)  # Use tracked task to prevent loss on shutdown
                return

            except Exception as e:
                logger.error(f"Sync request {request.request_id} failed: {e}")
                if self._metrics:
                    self._metrics.record_failure()
                self._resolve_future(
                    request.request_id,
                    LLMResponse(
                        request_id=request.request_id,
                        content=None,
                        success=False,
                        error=str(e),
                    ),
                )

    def _resolve_future(self, request_id: str, response: LLMResponse) -> None:
        """Resolve the future for a request.

        Args:
            request_id: ID of the request
            response: The response to resolve with
        """
        future = self._pending_futures.pop(request_id, None)
        if future and not future.done():
            future.set_result(response)

    # Batch monitoring

    async def _batch_monitor_loop(self) -> None:
        """Background loop to monitor submitted batches and periodically flush queued requests."""
        coordinator = get_shutdown_coordinator()
        last_flush_time = datetime.now(timezone.utc)

        while not coordinator.shutdown_requested:
            try:
                await self._check_submitted_batches()
            except Exception as e:
                logger.exception(f"Error in batch monitor loop: {e}")

            # Periodic flush: submit any queued requests that haven't been
            # picked up by a batch_group exit or threshold trigger
            elapsed = (datetime.now(timezone.utc) - last_flush_time).total_seconds()
            if elapsed >= self._config.flush_interval_seconds:
                try:
                    queue_size = await self._persistence.get_queue_size()
                    if queue_size > 0:
                        logger.info(f"Periodic flush: submitting {queue_size} queued requests")
                        await self.flush()
                    last_flush_time = datetime.now(timezone.utc)
                except Exception as e:
                    logger.exception(f"Error in periodic flush: {e}")

            # Wait or respond to shutdown
            if await coordinator.wait_or_shutdown(self._config.poll_interval_seconds):
                break

    async def _check_submitted_batches(self) -> None:
        """Check status of all submitted batches."""
        if not self._async_client:
            return

        batches = await self._persistence.get_submitted_batches()

        for batch_id, batch_info in batches.items():
            try:
                # Retrieve batch status
                batch = await self._async_client.messages.batches.retrieve(batch_id)

                if batch.processing_status == "ended":
                    await self._handle_batch_completed(batch_id, batch)
                else:
                    # Check for timeout
                    submitted_at = datetime.fromisoformat(batch_info["submitted_at"])
                    elapsed_hours = (datetime.now(timezone.utc) - submitted_at).total_seconds() / 3600

                    # Get timeout for current mode
                    retry_count = batch_info.get("retry_count", 0)
                    timeout_hours = self._config.get_wait_timeout_hours(self._mode, retry_count)

                    if elapsed_hours > timeout_hours:
                        await self._handle_batch_timeout(batch_id, batch_info)

            except Exception as e:
                logger.error(f"Error checking batch {batch_id}: {e}")

    async def _handle_batch_completed(
        self,
        batch_id: str,
        batch: Any,
    ) -> None:
        """Handle a completed batch.

        Args:
            batch_id: The batch ID
            batch: Anthropic batch object
        """
        logger.info(f"Batch {batch_id} completed, fetching results")

        # Calculate wait time for metrics
        batches = await self._persistence.get_submitted_batches()
        if batch_id in batches:
            submitted_at = datetime.fromisoformat(batches[batch_id]["submitted_at"])
            wait_seconds = (datetime.now(timezone.utc) - submitted_at).total_seconds()
            if self._metrics:
                self._metrics.record_batch_completed(wait_seconds)

        # Fetch results (mutates self._id_mapping to clean up processed entries)
        results = await fetch_batch_results(batch.results_url, self._id_mapping)

        # Update persistence and resolve futures
        completed_requests = await self._persistence.mark_batch_completed(batch_id, results)

        for request in completed_requests:
            result = results.get(request.request_id, {})
            self._resolve_future(
                request.request_id,
                LLMResponse(
                    request_id=request.request_id,
                    content=result.get("content"),
                    success=result.get("success", False),
                    error=result.get("error"),
                    usage=result.get("usage"),
                    model=result.get("model"),
                    stop_reason=result.get("stop_reason"),
                    thinking=result.get("thinking"),
                    content_blocks=result.get("content_blocks"),
                    batched=True,
                ),
            )

    def _validate_results_url(self, url: str) -> bool:
        """Validate that a results URL points to an allowed Anthropic domain.

        Delegates to result_fetcher.validate_results_url(). Preserved as
        instance method for test compatibility.
        """
        from .result_fetcher import validate_results_url
        return validate_results_url(url)

    async def _fetch_batch_results(
        self,
        results_url: str,
    ) -> dict[str, dict[str, Any]]:
        """Fetch and parse batch results from Anthropic.

        Delegates to result_fetcher.fetch_batch_results(). Preserved as
        instance method for test compatibility.
        """
        return await fetch_batch_results(results_url, self._id_mapping)

    # Batch retry and timeout

    async def _handle_batch_timeout(
        self,
        batch_id: str,
        batch_info: dict[str, Any],
    ) -> None:
        """Handle a batch that hasn't completed within timeout.

        Args:
            batch_id: The batch ID
            batch_info: Batch tracking info
        """
        request_ids = batch_info.get("request_ids", [])
        retry_count = batch_info.get("retry_count", 0)
        max_retries = self._config.max_retries_for_mode(self._mode)

        if self._metrics:
            self._metrics.record_batch_timeout()

        if retry_count >= max_retries:
            # Exhausted retries - fall back to sync
            logger.warning(
                f"Batch {batch_id} exhausted retries ({retry_count}/{max_retries}), "
                f"falling back to sync for {len(request_ids)} requests"
            )
            if self._metrics:
                self._metrics.record_sync_fallback()

            # Collect requests and modify queue atomically within single lock context
            # to avoid nested lock acquisition (remove_request also acquires lock)
            sync_requests: list[LLMRequest] = []
            request_ids_set = set(request_ids)

            async with self._persistence.lock():
                queue = await self._persistence.read_queue()

                # Separate requests to sync from those to keep
                remaining_requests = []
                for request in queue["requests"]:
                    if request["request_id"] in request_ids_set:
                        sync_requests.append(LLMRequest.from_dict(request))
                    else:
                        remaining_requests.append(request)

                # Update queue atomically: remove synced requests and batch
                queue["requests"] = remaining_requests
                queue["batches"].pop(batch_id, None)
                await self._persistence.write_queue(queue)

            # Execute sync tasks OUTSIDE the lock to avoid holding lock during I/O
            for req in sync_requests:
                self._spawn_sync_task(req)

        else:
            # Requeue for retry
            logger.info(f"Batch {batch_id} timed out, requeueing (retry {retry_count + 1}/{max_retries})")

            # Cancel the old batch on Anthropic's side to avoid duplicate processing
            try:
                await self._async_client.messages.batches.cancel(batch_id)
                logger.info(f"Cancelled batch {batch_id} on Anthropic")
            except Exception as e:
                logger.warning(f"Failed to cancel batch {batch_id} on Anthropic: {e}")

            async with self._persistence.lock():
                queue = await self._persistence.read_queue()

                # Increment retry counts and reset state
                for request in queue["requests"]:
                    if request["request_id"] in request_ids:
                        request["retry_count"] = request.get("retry_count", 0) + 1
                        request["state"] = RequestState.QUEUED.value
                        request["batch_id"] = None
                        request["submitted_at"] = None

                # Remove old batch entry - requests will be resubmitted in a new batch
                queue["batches"].pop(batch_id, None)

                await self._persistence.write_queue(queue)


# Global broker instance
_broker: LLMBroker | None = None


def get_broker() -> LLMBroker:
    """Get or create the global broker instance."""
    global _broker
    if _broker is None:
        _broker = LLMBroker()
    return _broker


def set_broker(broker: LLMBroker) -> None:
    """Set the global broker instance (useful for testing)."""
    global _broker
    _broker = broker


async def reset_broker() -> None:
    """Reset the global broker instance."""
    global _broker
    if _broker is not None:
        await _broker.stop()
    _broker = None
