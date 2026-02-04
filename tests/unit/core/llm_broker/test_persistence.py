"""Unit tests for LLM Broker persistence."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from core.llm_broker.persistence import BrokerPersistence
from core.llm_broker.schemas import (
    BatchPolicy,
    LLMRequest,
    RequestState,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def persistence(temp_dir):
    """Create a BrokerPersistence instance with temp directory."""
    return BrokerPersistence(temp_dir / "llm_broker")


class TestBrokerPersistence:
    """Tests for BrokerPersistence class."""

    @pytest.mark.asyncio
    async def test_initialize_creates_directory(self, persistence, temp_dir):
        """Test initialize creates queue directory and file."""
        await persistence.initialize()

        assert persistence.queue_dir.exists()
        assert persistence.queue_file.exists()

    @pytest.mark.asyncio
    async def test_initialize_creates_empty_queue(self, persistence):
        """Test initialize creates empty queue structure."""
        await persistence.initialize()

        queue = await persistence.read_queue()

        assert "requests" in queue
        assert "batches" in queue
        assert "last_updated" in queue
        assert queue["requests"] == []
        assert queue["batches"] == {}

    @pytest.mark.asyncio
    async def test_add_request(self, persistence):
        """Test adding a request to the queue."""
        await persistence.initialize()

        request = LLMRequest.create(
            prompt="Test prompt",
            model="claude-sonnet-4-5-20250929",
            policy=BatchPolicy.PREFER_SPEED,
        )
        await persistence.add_request(request)

        queue = await persistence.read_queue()
        assert len(queue["requests"]) == 1
        assert queue["requests"][0]["request_id"] == request.request_id
        assert queue["requests"][0]["prompt"] == "Test prompt"

    @pytest.mark.asyncio
    async def test_get_queued_requests(self, persistence):
        """Test getting queued requests."""
        await persistence.initialize()

        # Add multiple requests
        req1 = LLMRequest.create(prompt="Prompt 1", model="model-1")
        req2 = LLMRequest.create(prompt="Prompt 2", model="model-2")
        await persistence.add_request(req1)
        await persistence.add_request(req2)

        queued = await persistence.get_queued_requests()

        assert len(queued) == 2
        assert all(r.state == RequestState.QUEUED for r in queued)

    @pytest.mark.asyncio
    async def test_get_queued_requests_excludes_submitted(self, persistence):
        """Test get_queued_requests excludes non-QUEUED states."""
        await persistence.initialize()

        # Add request and mark as submitted
        request = LLMRequest.create(prompt="Test", model="model")
        await persistence.add_request(request)
        await persistence.mark_requests_submitted([request.request_id], "batch_123")

        # Should not include submitted requests
        queued = await persistence.get_queued_requests()
        assert len(queued) == 0

    @pytest.mark.asyncio
    async def test_mark_requests_submitted(self, persistence):
        """Test marking requests as submitted."""
        await persistence.initialize()

        request = LLMRequest.create(prompt="Test", model="model")
        await persistence.add_request(request)

        await persistence.mark_requests_submitted([request.request_id], "batch_456")

        queue = await persistence.read_queue()
        req_data = queue["requests"][0]

        assert req_data["state"] == RequestState.SUBMITTED.value
        assert req_data["batch_id"] == "batch_456"
        assert req_data["submitted_at"] is not None

        # Check batch tracking
        assert "batch_456" in queue["batches"]
        assert request.request_id in queue["batches"]["batch_456"]["request_ids"]

    @pytest.mark.asyncio
    async def test_mark_batch_completed(self, persistence):
        """Test marking a batch as completed."""
        await persistence.initialize()

        # Setup: add request and submit
        request = LLMRequest.create(prompt="Test", model="model")
        await persistence.add_request(request)
        await persistence.mark_requests_submitted([request.request_id], "batch_789")

        # Mark completed with results
        results = {
            request.request_id: {
                "success": True,
                "content": "Response content",
            }
        }
        completed = await persistence.mark_batch_completed("batch_789", results)

        assert len(completed) == 1
        assert completed[0].request_id == request.request_id
        assert completed[0].state == RequestState.COMPLETED

        # Batch should be removed
        queue = await persistence.read_queue()
        assert "batch_789" not in queue["batches"]

    @pytest.mark.asyncio
    async def test_mark_batch_completed_with_failures(self, persistence):
        """Test marking a batch with failed requests."""
        await persistence.initialize()

        # Setup
        request = LLMRequest.create(prompt="Test", model="model")
        await persistence.add_request(request)
        await persistence.mark_requests_submitted([request.request_id], "batch_fail")

        # Mark completed with failure
        results = {
            request.request_id: {
                "success": False,
                "error": "Rate limit",
            }
        }
        completed = await persistence.mark_batch_completed("batch_fail", results)

        assert completed[0].state == RequestState.FAILED

    @pytest.mark.asyncio
    async def test_increment_retry(self, persistence):
        """Test incrementing retry count."""
        await persistence.initialize()

        request = LLMRequest.create(prompt="Test", model="model")
        await persistence.add_request(request)

        new_count = await persistence.increment_retry(request.request_id)
        assert new_count == 1

        new_count = await persistence.increment_retry(request.request_id)
        assert new_count == 2

        # Check request state reset to QUEUED
        queued = await persistence.get_queued_requests()
        assert len(queued) == 1
        assert queued[0].retry_count == 2

    @pytest.mark.asyncio
    async def test_remove_request(self, persistence):
        """Test removing a request."""
        await persistence.initialize()

        request = LLMRequest.create(prompt="Test", model="model")
        await persistence.add_request(request)

        removed = await persistence.remove_request(request.request_id)

        assert removed is not None
        assert removed.request_id == request.request_id

        # Queue should be empty
        queue = await persistence.read_queue()
        assert len(queue["requests"]) == 0

    @pytest.mark.asyncio
    async def test_remove_request_not_found(self, persistence):
        """Test removing a non-existent request."""
        await persistence.initialize()

        removed = await persistence.remove_request("nonexistent")
        assert removed is None

    @pytest.mark.asyncio
    async def test_get_queue_size(self, persistence):
        """Test getting queue size."""
        await persistence.initialize()

        assert await persistence.get_queue_size() == 0

        # Add requests
        req1 = LLMRequest.create(prompt="1", model="m")
        req2 = LLMRequest.create(prompt="2", model="m")
        await persistence.add_request(req1)
        await persistence.add_request(req2)

        assert await persistence.get_queue_size() == 2

        # Submit one - should not count
        await persistence.mark_requests_submitted([req1.request_id], "batch")

        assert await persistence.get_queue_size() == 1

    @pytest.mark.asyncio
    async def test_cleanup_completed(self, persistence):
        """Test cleaning up old completed requests."""
        await persistence.initialize()

        # Add and complete a request
        request = LLMRequest.create(prompt="Test", model="model")
        await persistence.add_request(request)
        await persistence.mark_requests_submitted([request.request_id], "batch")
        await persistence.mark_batch_completed("batch", {request.request_id: {"success": True}})

        # With max_age_hours=0, should clean up immediately
        cleaned = await persistence.cleanup_completed(max_age_hours=0)

        assert cleaned == 1
        queue = await persistence.read_queue()
        assert len(queue["requests"]) == 0

    @pytest.mark.asyncio
    async def test_cleanup_completed_keeps_queued(self, persistence):
        """Test cleanup doesn't remove queued requests."""
        await persistence.initialize()

        request = LLMRequest.create(prompt="Test", model="model")
        await persistence.add_request(request)

        cleaned = await persistence.cleanup_completed(max_age_hours=0)

        assert cleaned == 0
        queue = await persistence.read_queue()
        assert len(queue["requests"]) == 1

    @pytest.mark.asyncio
    async def test_atomic_write(self, persistence, temp_dir):
        """Test that writes are atomic (temp file + rename)."""
        await persistence.initialize()

        # Add a request
        request = LLMRequest.create(prompt="Test", model="model")
        await persistence.add_request(request)

        # Verify queue file exists and is valid JSON
        with open(persistence.queue_file, "r") as f:
            data = json.load(f)
            assert len(data["requests"]) == 1

        # No temp file should remain
        temp_file = persistence.queue_file.with_suffix(".tmp")
        assert not temp_file.exists()

    @pytest.mark.asyncio
    async def test_concurrent_access(self, persistence):
        """Test concurrent access to the queue."""
        await persistence.initialize()

        async def add_request(i: int):
            request = LLMRequest.create(prompt=f"Prompt {i}", model="model")
            await persistence.add_request(request)

        # Add multiple requests concurrently
        await asyncio.gather(*[add_request(i) for i in range(10)])

        queue = await persistence.read_queue()
        assert len(queue["requests"]) == 10


class TestBrokerPersistenceLocking:
    """Tests for file locking behavior."""

    @pytest.mark.asyncio
    async def test_lock_context_manager(self, persistence):
        """Test lock context manager acquires and releases lock."""
        await persistence.initialize()

        async with persistence.lock():
            # Should be able to read/write while holding lock
            queue = await persistence.read_queue()
            assert queue is not None

        # Lock should be released

    @pytest.mark.asyncio
    async def test_lock_file_created(self, persistence):
        """Test lock file is created."""
        await persistence.initialize()

        async with persistence.lock():
            assert persistence.lock_file.exists()
