"""Unit tests for LLM Broker core functionality."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.llm_broker.broker import LLMBroker, _sanitize_custom_id
from core.llm_broker.config import BrokerConfig
from core.llm_broker.exceptions import (
    QueueOverflowError,
)
from core.llm_broker.schemas import (
    BatchPolicy,
    UserMode,
)
from workflows.shared.llm_utils import ModelTier


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration with temp queue directory."""
    return BrokerConfig(
        default_mode=UserMode.BALANCED,
        batch_threshold=5,
        max_queue_size=10,
        queue_dir=str(temp_dir / "llm_broker"),
        poll_interval_seconds=1,  # Fast polling for tests
    )


@pytest.fixture
async def broker(test_config):
    """Create and start a broker for testing."""
    from core.task_queue.shutdown import reset_shutdown_coordinator

    # Reset shutdown coordinator to ensure clean state
    reset_shutdown_coordinator()

    broker = LLMBroker(config=test_config)
    # Mock the Anthropic client
    broker._async_client = MagicMock()

    # Don't start the background monitor task for tests - it causes hangs
    # Instead, manually initialize persistence
    await broker._persistence.initialize()
    broker._started = True

    yield broker

    broker._started = False
    reset_shutdown_coordinator()


class TestSanitizeCustomId:
    """Tests for custom ID sanitization."""

    def test_valid_id_unchanged(self):
        """Test valid IDs are unchanged."""
        assert _sanitize_custom_id("abc123") == "abc123"
        assert _sanitize_custom_id("test_id-123") == "test_id-123"

    def test_special_chars_replaced(self):
        """Test special characters are replaced with underscore."""
        assert _sanitize_custom_id("test.id") == "test_id"
        assert _sanitize_custom_id("test:id") == "test_id"
        assert _sanitize_custom_id("test/id") == "test_id"
        assert _sanitize_custom_id("test@id#123") == "test_id_123"

    def test_truncation_at_64_chars(self):
        """Test IDs longer than 64 chars are truncated."""
        long_id = "a" * 100
        result = _sanitize_custom_id(long_id)
        assert len(result) == 64
        assert result == "a" * 64

    def test_uuid_sanitization(self):
        """Test UUID-style IDs are sanitized."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = _sanitize_custom_id(uuid)
        assert result == uuid  # UUID chars are all valid


class TestBrokerLifecycle:
    """Tests for broker start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_initializes_persistence(self, test_config):
        """Test start() initializes persistence."""
        from core.task_queue.shutdown import get_shutdown_coordinator, reset_shutdown_coordinator

        reset_shutdown_coordinator()
        coordinator = get_shutdown_coordinator()

        broker = LLMBroker(config=test_config)
        broker._async_client = MagicMock()

        await broker.start()

        assert broker._started is True
        assert Path(test_config.queue_dir).exists()

        # Signal shutdown to allow background task to exit
        coordinator.request_shutdown()
        await broker.stop()
        reset_shutdown_coordinator()

    @pytest.mark.asyncio
    async def test_stop_cancels_background_task(self, test_config):
        """Test stop() cancels background monitor task."""
        from core.task_queue.shutdown import get_shutdown_coordinator, reset_shutdown_coordinator

        reset_shutdown_coordinator()
        coordinator = get_shutdown_coordinator()

        broker = LLMBroker(config=test_config)
        broker._async_client = MagicMock()

        await broker.start()
        assert broker._batch_monitor_task is not None

        coordinator.request_shutdown()
        await broker.stop()
        assert broker._started is False
        reset_shutdown_coordinator()

    @pytest.mark.asyncio
    async def test_request_before_start_auto_starts(self, test_config):
        """Test making request before start() auto-starts the broker."""
        from core.task_queue.shutdown import get_shutdown_coordinator, reset_shutdown_coordinator

        reset_shutdown_coordinator()
        coordinator = get_shutdown_coordinator()

        broker = LLMBroker(config=test_config)
        broker._async_client = MagicMock()
        assert not broker._started

        # Mock _queue_for_batch and _spawn_sync_task so request() doesn't
        # actually try to call the API
        broker._spawn_sync_task = MagicMock()

        await broker.request(prompt="Test", model=ModelTier.SONNET)
        assert broker._started

        coordinator.request_shutdown()
        await broker.stop()
        reset_shutdown_coordinator()


class TestRoutingLogic:
    """Tests for request routing logic."""

    @pytest.mark.asyncio
    async def test_fast_mode_never_batches(self, test_config):
        """Test FAST mode always uses sync API."""
        test_config.default_mode = UserMode.FAST
        broker = LLMBroker(config=test_config, mode=UserMode.FAST)

        # Test all policies except FORCE_BATCH
        assert broker._should_batch(BatchPolicy.PREFER_SPEED, ModelTier.SONNET, None) is False
        assert broker._should_batch(BatchPolicy.PREFER_BALANCE, ModelTier.SONNET, None) is False
        assert broker._should_batch(BatchPolicy.REQUIRE_SYNC, ModelTier.SONNET, None) is False

    @pytest.mark.asyncio
    async def test_force_batch_always_batches(self, test_config):
        """Test FORCE_BATCH always batches regardless of mode."""
        broker = LLMBroker(config=test_config, mode=UserMode.FAST)

        # FORCE_BATCH should batch even in FAST mode
        assert broker._should_batch(BatchPolicy.FORCE_BATCH, ModelTier.SONNET, None) is True

    @pytest.mark.asyncio
    async def test_require_sync_never_batches(self, test_config):
        """Test REQUIRE_SYNC never batches."""
        broker = LLMBroker(config=test_config, mode=UserMode.ECONOMICAL)

        # REQUIRE_SYNC should not batch even in ECONOMICAL mode
        assert broker._should_batch(BatchPolicy.REQUIRE_SYNC, ModelTier.SONNET, None) is False

    @pytest.mark.asyncio
    async def test_balanced_mode_routing(self, test_config):
        """Test BALANCED mode routing logic."""
        broker = LLMBroker(config=test_config, mode=UserMode.BALANCED)

        # PREFER_BALANCE should batch in BALANCED mode
        assert broker._should_batch(BatchPolicy.PREFER_BALANCE, ModelTier.SONNET, None) is True

        # PREFER_SPEED should NOT batch in BALANCED mode
        assert broker._should_batch(BatchPolicy.PREFER_SPEED, ModelTier.SONNET, None) is False

    @pytest.mark.asyncio
    async def test_economical_mode_routing(self, test_config):
        """Test ECONOMICAL mode routing logic."""
        broker = LLMBroker(config=test_config, mode=UserMode.ECONOMICAL)

        # All batch-eligible policies should batch in ECONOMICAL mode
        assert broker._should_batch(BatchPolicy.PREFER_BALANCE, ModelTier.SONNET, None) is True
        assert broker._should_batch(BatchPolicy.PREFER_SPEED, ModelTier.SONNET, None) is True

    @pytest.mark.asyncio
    async def test_deepseek_never_batches(self, test_config):
        """Test DeepSeek models never batch."""
        broker = LLMBroker(config=test_config, mode=UserMode.ECONOMICAL)

        assert broker._should_batch(BatchPolicy.PREFER_BALANCE, ModelTier.DEEPSEEK_V3, None) is False
        assert broker._should_batch(BatchPolicy.PREFER_BALANCE, ModelTier.DEEPSEEK_R1, None) is False

    @pytest.mark.asyncio
    async def test_extended_thinking_never_batches(self, test_config):
        """Test extended thinking requests never batch."""
        broker = LLMBroker(config=test_config, mode=UserMode.ECONOMICAL)

        # With thinking_budget set, should not batch
        assert broker._should_batch(BatchPolicy.PREFER_BALANCE, ModelTier.SONNET, thinking_budget=4000) is False


class TestBatchGroup:
    """Tests for batch_group context manager."""

    @pytest.mark.asyncio
    async def test_batch_group_tracks_requests(self, broker):
        """Test batch_group tracks request IDs."""
        # Mock _submit_batch to avoid actual API calls
        broker._submit_batch = AsyncMock()

        async with broker.batch_group() as group:
            # Add requests with FORCE_BATCH to ensure they queue
            await broker.request(
                prompt="Test 1",
                model=ModelTier.SONNET,
                policy=BatchPolicy.FORCE_BATCH,
            )
            await broker.request(
                prompt="Test 2",
                model=ModelTier.SONNET,
                policy=BatchPolicy.FORCE_BATCH,
            )

            assert len(group.request_ids) == 2

        # Verify _submit_batch was called on context exit
        assert broker._submit_batch.called

    @pytest.mark.asyncio
    async def test_batch_group_with_mode_override(self, broker):
        """Test batch_group can override mode."""
        broker._submit_batch = AsyncMock()  # Mock to avoid API calls
        async with broker.batch_group(mode=UserMode.FAST) as group:
            assert group.mode == UserMode.FAST


class TestQueueOverflow:
    """Tests for queue overflow protection."""

    @pytest.mark.asyncio
    async def test_overflow_sync_fallback(self, test_config):
        """Test overflow with sync fallback behavior."""
        from core.task_queue.shutdown import reset_shutdown_coordinator

        reset_shutdown_coordinator()
        test_config.max_queue_size = 2
        test_config.overflow_behavior = "sync"

        broker = LLMBroker(config=test_config)
        broker._async_client = MagicMock()
        broker._execute_sync = AsyncMock()

        # Initialize without background task
        await broker._persistence.initialize()
        broker._started = True

        try:
            # Fill the queue
            await broker.request(prompt="1", model=ModelTier.SONNET, policy=BatchPolicy.FORCE_BATCH)
            await broker.request(prompt="2", model=ModelTier.SONNET, policy=BatchPolicy.FORCE_BATCH)

            # Third request should fall back to sync
            await broker.request(prompt="3", model=ModelTier.SONNET, policy=BatchPolicy.FORCE_BATCH)

            # _execute_sync should have been called for the overflow
            assert broker._execute_sync.called
        finally:
            broker._started = False
            reset_shutdown_coordinator()

    @pytest.mark.asyncio
    async def test_overflow_reject(self, test_config):
        """Test overflow with reject behavior."""
        from core.task_queue.shutdown import reset_shutdown_coordinator

        reset_shutdown_coordinator()
        test_config.max_queue_size = 2
        test_config.overflow_behavior = "reject"

        broker = LLMBroker(config=test_config)
        broker._async_client = MagicMock()

        # Initialize without background task
        await broker._persistence.initialize()
        broker._started = True

        try:
            # Fill the queue
            await broker.request(prompt="1", model=ModelTier.SONNET, policy=BatchPolicy.FORCE_BATCH)
            await broker.request(prompt="2", model=ModelTier.SONNET, policy=BatchPolicy.FORCE_BATCH)

            # Third request should raise
            with pytest.raises(QueueOverflowError) as exc_info:
                await broker.request(prompt="3", model=ModelTier.SONNET, policy=BatchPolicy.FORCE_BATCH)

            assert exc_info.value.queue_size >= exc_info.value.max_size
        finally:
            broker._started = False
            reset_shutdown_coordinator()


class TestFutureResolution:
    """Tests for Future resolution mechanics."""

    @pytest.mark.asyncio
    async def test_future_resolved_on_sync_success(self, broker):
        """Test Future is resolved when sync request succeeds."""
        # Mock successful sync response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.stop_reason = "end_turn"

        # Mock both regular and beta paths (SONNET and SONNET_1M have same model value)
        broker._async_client.messages.create = AsyncMock(return_value=mock_response)
        broker._async_client.beta.messages.create = AsyncMock(return_value=mock_response)

        future = await broker.request(
            prompt="Test",
            model=ModelTier.HAIKU,  # Use HAIKU to avoid SONNET_1M check
            policy=BatchPolicy.REQUIRE_SYNC,  # Force sync path
        )

        # Wait for the sync task to complete
        response = await asyncio.wait_for(future, timeout=5.0)

        assert response.success is True
        assert response.content == "Response"

    @pytest.mark.asyncio
    async def test_future_resolved_on_sync_failure(self, broker):
        """Test Future is resolved with error on sync failure."""
        # Mock both paths
        broker._async_client.messages.create = AsyncMock(side_effect=Exception("API Error"))
        broker._async_client.beta.messages.create = AsyncMock(side_effect=Exception("API Error"))

        future = await broker.request(
            prompt="Test",
            model=ModelTier.HAIKU,  # Use HAIKU to avoid SONNET_1M check
            policy=BatchPolicy.REQUIRE_SYNC,
        )

        response = await asyncio.wait_for(future, timeout=5.0)

        assert response.success is False
        assert "API Error" in response.error


class TestMetrics:
    """Tests for metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_recorded_for_sync(self, broker):
        """Test metrics are recorded for sync requests."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_response.model = "test"
        mock_response.stop_reason = "end_turn"

        broker._async_client.messages.create = AsyncMock(return_value=mock_response)
        broker._async_client.beta.messages.create = AsyncMock(return_value=mock_response)

        await broker.request(
            prompt="Test",
            model=ModelTier.HAIKU,  # Use HAIKU to avoid SONNET_1M check
            policy=BatchPolicy.REQUIRE_SYNC,
        )

        # Give task time to complete
        await asyncio.sleep(0.1)

        metrics = broker.metrics.to_dict()
        assert metrics["requests_total"] >= 1
        assert metrics["requests_sync"] >= 1

    @pytest.mark.asyncio
    async def test_metrics_recorded_for_batch(self, broker):
        """Test metrics are recorded for batch requests."""
        await broker.request(
            prompt="Test",
            model=ModelTier.SONNET,
            policy=BatchPolicy.FORCE_BATCH,
        )

        metrics = broker.metrics.to_dict()
        assert metrics["requests_total"] >= 1
        assert metrics["requests_batched"] >= 1


class TestModeProperty:
    """Tests for mode property."""

    @pytest.mark.asyncio
    async def test_get_mode(self, broker):
        """Test getting current mode."""
        assert broker.mode == UserMode.BALANCED

    @pytest.mark.asyncio
    async def test_set_mode(self, broker):
        """Test setting mode."""
        broker.mode = UserMode.ECONOMICAL
        assert broker.mode == UserMode.ECONOMICAL


class TestSyncTaskTracking:
    """Tests for fire-and-forget task tracking and graceful shutdown."""

    @pytest.mark.asyncio
    async def test_sync_tasks_tracked_in_set(self, broker):
        """Test that sync execution tasks are tracked in _sync_tasks set."""
        # Create a slow mock response to keep the task running
        async def slow_create(**kwargs):
            await asyncio.sleep(0.5)
            mock_response = MagicMock()
            mock_response.content = [MagicMock(type="text", text="Response")]
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
            mock_response.model = "test"
            mock_response.stop_reason = "end_turn"
            return mock_response

        broker._async_client.messages.create = slow_create
        broker._async_client.beta.messages.create = slow_create

        # Make a sync request
        await broker.request(
            prompt="Test",
            model=ModelTier.HAIKU,
            policy=BatchPolicy.REQUIRE_SYNC,
        )

        # Task should be tracked while running
        await asyncio.sleep(0.1)  # Give task time to start
        assert len(broker._sync_tasks) >= 1

        # Wait for completion and verify task is removed
        await asyncio.sleep(0.6)
        assert len(broker._sync_tasks) == 0

    @pytest.mark.asyncio
    async def test_stop_waits_for_in_flight_sync_tasks(self, test_config):
        """Test that stop() waits for all in-flight sync tasks to complete."""
        from core.task_queue.shutdown import reset_shutdown_coordinator

        reset_shutdown_coordinator()

        broker = LLMBroker(config=test_config)
        broker._async_client = MagicMock()

        # Track whether task completed before stop returned
        task_completed = False

        async def slow_create(**kwargs):
            nonlocal task_completed
            await asyncio.sleep(0.3)
            task_completed = True
            mock_response = MagicMock()
            mock_response.content = [MagicMock(type="text", text="Response")]
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
            mock_response.model = "test"
            mock_response.stop_reason = "end_turn"
            return mock_response

        broker._async_client.messages.create = slow_create
        broker._async_client.beta.messages.create = slow_create

        # Initialize without background task to avoid shutdown coordinator issues
        await broker._persistence.initialize()
        broker._started = True

        try:
            # Submit a sync request
            await broker.request(
                prompt="Test",
                model=ModelTier.HAIKU,
                policy=BatchPolicy.REQUIRE_SYNC,
            )

            # Verify task is in-flight
            await asyncio.sleep(0.1)
            assert len(broker._sync_tasks) >= 1

            # Stop should wait for the task
            await broker.stop()

            # Task should have completed before stop returned
            assert task_completed is True
            assert len(broker._sync_tasks) == 0
        finally:
            broker._started = False
            reset_shutdown_coordinator()

    @pytest.mark.asyncio
    async def test_task_exceptions_logged_not_orphaned(self, broker, caplog):
        """Test that task exceptions are logged via done callback."""
        import logging

        # Make the API call raise an exception
        broker._async_client.messages.create = AsyncMock(
            side_effect=Exception("Simulated API failure")
        )
        broker._async_client.beta.messages.create = AsyncMock(
            side_effect=Exception("Simulated API failure")
        )

        with caplog.at_level(logging.ERROR):
            await broker.request(
                prompt="Test",
                model=ModelTier.HAIKU,
                policy=BatchPolicy.REQUIRE_SYNC,
            )

            # Wait for task to complete and callback to run
            await asyncio.sleep(0.2)

        # The error should be logged (either from _execute_sync or done callback)
        assert any("failed" in record.message.lower() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_spawn_sync_task_returns_named_task(self, broker):
        """Test that _spawn_sync_task creates properly named tasks."""
        from core.llm_broker.schemas import LLMRequest

        # Create a test request
        request = LLMRequest.create(
            prompt="Test",
            model=ModelTier.HAIKU.value,
            policy=BatchPolicy.REQUIRE_SYNC,
            max_tokens=100,
        )

        # Mock to avoid actual API call
        broker._async_client.messages.create = AsyncMock(
            side_effect=Exception("Expected")
        )

        task = broker._spawn_sync_task(request)

        # Task should be named with request ID prefix
        assert task.get_name().startswith("sync-")
        assert request.request_id[:8] in task.get_name()

        # Wait for task to complete
        await asyncio.sleep(0.1)


class TestResultsUrlValidation:
    """Tests for SSRF protection via results URL validation."""

    @pytest.mark.asyncio
    async def test_valid_api_anthropic_url(self, broker):
        """Test that api.anthropic.com HTTPS URLs are accepted."""
        url = "https://api.anthropic.com/v1/batches/123/results"
        assert broker._validate_results_url(url) is True

    @pytest.mark.asyncio
    async def test_valid_batches_anthropic_url(self, broker):
        """Test that batches.anthropic.com HTTPS URLs are accepted."""
        url = "https://batches.anthropic.com/results/abc123"
        assert broker._validate_results_url(url) is True

    @pytest.mark.asyncio
    async def test_reject_http_url(self, broker):
        """Test that HTTP URLs are rejected even for valid domains."""
        url = "http://api.anthropic.com/v1/batches/123/results"
        assert broker._validate_results_url(url) is False

    @pytest.mark.asyncio
    async def test_reject_arbitrary_domain(self, broker):
        """Test that arbitrary domains are rejected."""
        url = "https://evil.com/steal-api-key"
        assert broker._validate_results_url(url) is False

    @pytest.mark.asyncio
    async def test_reject_subdomain_attack(self, broker):
        """Test that subdomains of allowed hosts are rejected."""
        url = "https://evil.api.anthropic.com/results"
        assert broker._validate_results_url(url) is False

    @pytest.mark.asyncio
    async def test_reject_lookalike_domain(self, broker):
        """Test that lookalike domains are rejected."""
        url = "https://api-anthropic.com/results"
        assert broker._validate_results_url(url) is False
        url = "https://anthropic.com.evil.com/results"
        assert broker._validate_results_url(url) is False

    @pytest.mark.asyncio
    async def test_reject_empty_url(self, broker):
        """Test that empty URLs are rejected."""
        assert broker._validate_results_url("") is False

    @pytest.mark.asyncio
    async def test_reject_malformed_url(self, broker):
        """Test that malformed URLs are rejected."""
        assert broker._validate_results_url("not-a-url") is False
        assert broker._validate_results_url("://missing-scheme.com") is False

    @pytest.mark.asyncio
    async def test_fetch_batch_results_validates_url(self, broker):
        """Test that _fetch_batch_results raises ValueError for invalid URLs."""
        with pytest.raises(ValueError) as exc_info:
            await broker._fetch_batch_results("https://evil.com/steal")

        assert "Invalid batch results URL" in str(exc_info.value)
        assert "evil.com" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_batch_results_error_includes_allowed_hosts(self, broker):
        """Test that error message lists allowed hosts for debugging."""
        with pytest.raises(ValueError) as exc_info:
            await broker._fetch_batch_results("https://attacker.com/results")

        error_msg = str(exc_info.value)
        assert "api.anthropic.com" in error_msg
        assert "batches.anthropic.com" in error_msg


class TestIdMappingCleanup:
    """Tests for _id_mapping cleanup to prevent memory leaks."""

    @pytest.mark.asyncio
    async def test_id_mapping_cleaned_after_batch_results_fetched(self, broker):
        """Test that _id_mapping entries are removed after processing batch results."""
        import httpx

        # Pre-populate _id_mapping with sanitized -> original mappings
        broker._id_mapping["sanitized_id_1"] = "original-id-1"
        broker._id_mapping["sanitized_id_2"] = "original-id-2"
        broker._id_mapping["unrelated_id"] = "should-remain"

        # Mock the HTTP response with JSONL results
        jsonl_response = (
            '{"custom_id": "sanitized_id_1", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": "Response 1"}]}}}\n'
            '{"custom_id": "sanitized_id_2", "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": "Response 2"}]}}}'
        )

        mock_response = MagicMock()
        mock_response.text = jsonl_response
        mock_response.raise_for_status = MagicMock()

        # Patch httpx.AsyncClient
        original_get = httpx.AsyncClient.get

        async def mock_get(self, url, **kwargs):
            return mock_response

        httpx.AsyncClient.get = mock_get

        try:
            results = await broker._fetch_batch_results(
                "https://api.anthropic.com/v1/batches/123/results"
            )

            # Verify results were parsed correctly
            assert "original-id-1" in results
            assert "original-id-2" in results
            assert results["original-id-1"]["success"] is True
            assert results["original-id-2"]["success"] is True

            # Verify processed IDs were cleaned up
            assert "sanitized_id_1" not in broker._id_mapping
            assert "sanitized_id_2" not in broker._id_mapping

            # Verify unrelated entry remains
            assert broker._id_mapping.get("unrelated_id") == "should-remain"
        finally:
            httpx.AsyncClient.get = original_get

    @pytest.mark.asyncio
    async def test_id_mapping_cleaned_for_errored_results(self, broker):
        """Test that _id_mapping entries are cleaned up even for errored results."""
        import httpx

        broker._id_mapping["error_id"] = "original-error-id"

        jsonl_response = '{"custom_id": "error_id", "result": {"type": "errored", "error": {"type": "invalid_request", "message": "Test error"}}}'

        mock_response = MagicMock()
        mock_response.text = jsonl_response
        mock_response.raise_for_status = MagicMock()

        original_get = httpx.AsyncClient.get

        async def mock_get(self, url, **kwargs):
            return mock_response

        httpx.AsyncClient.get = mock_get

        try:
            results = await broker._fetch_batch_results(
                "https://api.anthropic.com/v1/batches/123/results"
            )

            # Verify error result was parsed
            assert "original-error-id" in results
            assert results["original-error-id"]["success"] is False

            # Verify ID mapping was cleaned up
            assert "error_id" not in broker._id_mapping
        finally:
            httpx.AsyncClient.get = original_get

    @pytest.mark.asyncio
    async def test_id_mapping_cleaned_for_canceled_results(self, broker):
        """Test that _id_mapping entries are cleaned up for canceled/expired results."""
        import httpx

        broker._id_mapping["canceled_id"] = "original-canceled-id"

        jsonl_response = '{"custom_id": "canceled_id", "result": {"type": "canceled"}}'

        mock_response = MagicMock()
        mock_response.text = jsonl_response
        mock_response.raise_for_status = MagicMock()

        original_get = httpx.AsyncClient.get

        async def mock_get(self, url, **kwargs):
            return mock_response

        httpx.AsyncClient.get = mock_get

        try:
            results = await broker._fetch_batch_results(
                "https://api.anthropic.com/v1/batches/123/results"
            )

            # Verify canceled result was parsed
            assert "original-canceled-id" in results
            assert results["original-canceled-id"]["success"] is False

            # Verify ID mapping was cleaned up
            assert "canceled_id" not in broker._id_mapping
        finally:
            httpx.AsyncClient.get = original_get
