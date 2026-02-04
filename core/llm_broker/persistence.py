"""Async-safe queue persistence with file locking.

Provides cross-process coordination for the broker queue using
fcntl file locking wrapped in asyncio.to_thread() to avoid
blocking the event loop.
"""

import asyncio
import fcntl
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from .schemas import LLMRequest, RequestState

logger = logging.getLogger(__name__)


class BrokerPersistence:
    """Async-safe file-based queue with cross-process coordination.

    Uses fcntl.flock for exclusive locking and atomic writes via
    temp file + rename pattern. All blocking I/O is wrapped in
    asyncio.to_thread() to avoid blocking the event loop.
    """

    def __init__(self, queue_dir: str | Path) -> None:
        """Initialize persistence handler.

        Args:
            queue_dir: Directory for queue and lock files
        """
        self.queue_dir = Path(queue_dir)
        self.queue_file = self.queue_dir / "queue.json"
        self.lock_file = self.queue_dir / "queue.lock"

    async def initialize(self) -> None:
        """Initialize the queue directory and files.

        Creates the directory and empty queue file if they don't exist.
        """
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        """Sync helper for initialization."""
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        if not self.queue_file.exists():
            self._write_queue_sync(self._empty_queue())
        logger.debug(f"Broker persistence initialized at {self.queue_dir}")

    @asynccontextmanager
    async def lock(self) -> AsyncIterator[None]:
        """Acquire exclusive lock WITHOUT blocking event loop.

        Uses asyncio.to_thread() to run blocking fcntl.flock() in
        a thread pool, allowing other async tasks to proceed.

        Yields:
            None (context manager)
        """
        lock_fd = await asyncio.to_thread(self._acquire_lock)
        try:
            yield
        finally:
            await asyncio.to_thread(self._release_lock, lock_fd)

    def _acquire_lock(self) -> Any:
        """Sync helper - runs in thread pool."""
        self.lock_file.touch(exist_ok=True)
        lock_fd = open(self.lock_file, "w")
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        return lock_fd

    def _release_lock(self, lock_fd: Any) -> None:
        """Sync helper - runs in thread pool."""
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()

    async def read_queue(self) -> dict[str, Any]:
        """Read queue without blocking event loop.

        Returns:
            Queue data dictionary with 'requests' and 'batches' keys
        """
        return await asyncio.to_thread(self._read_queue_sync)

    def _read_queue_sync(self) -> dict[str, Any]:
        """Sync helper for reading queue."""
        if not self.queue_file.exists():
            return self._empty_queue()
        try:
            with open(self.queue_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Failed to read queue file, returning empty queue")
            return self._empty_queue()

    async def write_queue(self, queue: dict[str, Any]) -> None:
        """Atomic write via temp file + rename, non-blocking.

        Args:
            queue: Queue data to persist
        """
        await asyncio.to_thread(self._write_queue_sync, queue)

    def _write_queue_sync(self, queue: dict[str, Any]) -> None:
        """Sync helper for atomic queue write."""
        queue["last_updated"] = datetime.now(timezone.utc).isoformat()
        temp_file = self.queue_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(queue, f, indent=2)
        try:
            temp_file.rename(self.queue_file)
        except FileNotFoundError:
            # Temp file may have been deleted by concurrent cleanup
            logger.warning(f"Temp file {temp_file} disappeared before rename - retrying write")
            with open(self.queue_file, "w") as f:
                json.dump(queue, f, indent=2)

    def _empty_queue(self) -> dict[str, Any]:
        """Create an empty queue structure."""
        return {
            "requests": [],  # List of LLMRequest.to_dict()
            "batches": {},  # batch_id -> {request_ids, submitted_at, status}
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    # High-level operations

    async def add_request(self, request: LLMRequest) -> None:
        """Add a request to the queue.

        Args:
            request: The request to queue
        """
        async with self.lock():
            queue = await self.read_queue()
            queue["requests"].append(request.to_dict())
            await self.write_queue(queue)
            logger.debug(f"Added request {request.request_id} to queue")

    async def get_queued_requests(self) -> list[LLMRequest]:
        """Get all requests in QUEUED state.

        Returns:
            List of queued requests
        """
        async with self.lock():
            queue = await self.read_queue()
            return [LLMRequest.from_dict(r) for r in queue["requests"] if r["state"] == RequestState.QUEUED.value]

    async def get_submitted_batches(self) -> dict[str, dict[str, Any]]:
        """Get all submitted batches awaiting response.

        Returns:
            Dictionary of batch_id -> batch info
        """
        async with self.lock():
            queue = await self.read_queue()
            return queue.get("batches", {})

    async def mark_requests_submitted(
        self,
        request_ids: list[str],
        batch_id: str,
    ) -> None:
        """Mark requests as submitted with a batch ID.

        Args:
            request_ids: IDs of requests to mark
            batch_id: Anthropic batch ID
        """
        async with self.lock():
            queue = await self.read_queue()
            now = datetime.now(timezone.utc).isoformat()

            # Update request states
            for request in queue["requests"]:
                if request["request_id"] in request_ids:
                    request["state"] = RequestState.SUBMITTED.value
                    request["submitted_at"] = now
                    request["batch_id"] = batch_id

            # Track batch
            queue["batches"][batch_id] = {
                "request_ids": request_ids,
                "submitted_at": now,
                "status": "submitted",
            }

            await self.write_queue(queue)
            logger.debug(f"Marked {len(request_ids)} requests as submitted in batch {batch_id}")

    async def mark_batch_completed(
        self,
        batch_id: str,
        results: dict[str, Any],
    ) -> list[LLMRequest]:
        """Mark a batch as completed and update request states.

        Args:
            batch_id: The completed batch ID
            results: Dictionary of request_id -> result data

        Returns:
            List of completed requests
        """
        async with self.lock():
            queue = await self.read_queue()
            completed_requests = []

            for request in queue["requests"]:
                if request.get("batch_id") == batch_id:
                    request_id = request["request_id"]
                    if request_id in results:
                        result = results[request_id]
                        if result.get("success", True):
                            request["state"] = RequestState.COMPLETED.value
                        else:
                            request["state"] = RequestState.FAILED.value
                    else:
                        # No result for this request - mark failed
                        request["state"] = RequestState.FAILED.value
                    completed_requests.append(LLMRequest.from_dict(request))

            # Remove batch from tracking
            if batch_id in queue["batches"]:
                del queue["batches"][batch_id]

            await self.write_queue(queue)
            logger.debug(f"Batch {batch_id} completed with {len(completed_requests)} requests")
            return completed_requests

    async def increment_retry(self, request_id: str) -> int:
        """Increment retry count for a request.

        Args:
            request_id: ID of the request to retry

        Returns:
            New retry count
        """
        async with self.lock():
            queue = await self.read_queue()
            for request in queue["requests"]:
                if request["request_id"] == request_id:
                    request["retry_count"] = request.get("retry_count", 0) + 1
                    request["state"] = RequestState.QUEUED.value
                    request["batch_id"] = None
                    request["submitted_at"] = None
                    await self.write_queue(queue)
                    return request["retry_count"]
            return 0

    async def remove_request(self, request_id: str) -> LLMRequest | None:
        """Remove a request from the queue.

        Args:
            request_id: ID of the request to remove

        Returns:
            The removed request, or None if not found
        """
        async with self.lock():
            queue = await self.read_queue()
            for i, request in enumerate(queue["requests"]):
                if request["request_id"] == request_id:
                    removed = queue["requests"].pop(i)
                    await self.write_queue(queue)
                    return LLMRequest.from_dict(removed)
            return None

    async def get_queue_size(self) -> int:
        """Get current queue size (QUEUED state only).

        Returns:
            Number of queued requests
        """
        async with self.lock():
            queue = await self.read_queue()
            return sum(1 for r in queue["requests"] if r["state"] == RequestState.QUEUED.value)

    async def cleanup_completed(self, max_age_hours: float = 24.0) -> int:
        """Remove completed/failed requests older than max_age.

        Args:
            max_age_hours: Maximum age in hours for completed requests

        Returns:
            Number of requests cleaned up
        """
        async with self.lock():
            queue = await self.read_queue()
            now = datetime.now(timezone.utc)
            original_count = len(queue["requests"])

            queue["requests"] = [r for r in queue["requests"] if not self._should_cleanup(r, now, max_age_hours)]

            cleaned = original_count - len(queue["requests"])
            if cleaned > 0:
                await self.write_queue(queue)
                logger.info(f"Cleaned up {cleaned} old requests from queue")
            return cleaned

    def _should_cleanup(
        self,
        request: dict[str, Any],
        now: datetime,
        max_age_hours: float,
    ) -> bool:
        """Check if a request should be cleaned up."""
        state = request.get("state")
        if state not in (RequestState.COMPLETED.value, RequestState.FAILED.value):
            return False

        created_at = datetime.fromisoformat(request["created_at"])
        age_hours = (now - created_at).total_seconds() / 3600
        return age_hours > max_age_hours
