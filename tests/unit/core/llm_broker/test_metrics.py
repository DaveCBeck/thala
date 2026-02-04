"""Unit tests for LLM Broker metrics."""

from collections import deque

from core.llm_broker.metrics import METRICS_HISTORY_MAXLEN, BrokerMetrics


class TestBrokerMetrics:
    """Tests for BrokerMetrics class."""

    def test_initial_values(self):
        """Test metrics initialize to zero."""
        metrics = BrokerMetrics()

        assert metrics.requests_total == 0
        assert metrics.requests_batched == 0
        assert metrics.requests_sync == 0
        assert metrics.requests_batch_timeout == 0
        assert metrics.requests_failed == 0
        assert metrics.batches_submitted == 0
        assert len(metrics.batch_sizes) == 0
        assert len(metrics.batch_wait_times) == 0
        assert metrics.queue_overflow_count == 0
        assert metrics.sync_fallback_count == 0

    def test_batch_sizes_is_bounded_deque(self):
        """Test batch_sizes uses bounded deque to prevent memory growth."""
        metrics = BrokerMetrics()

        assert isinstance(metrics.batch_sizes, deque)
        assert metrics.batch_sizes.maxlen == METRICS_HISTORY_MAXLEN

    def test_batch_wait_times_is_bounded_deque(self):
        """Test batch_wait_times uses bounded deque to prevent memory growth."""
        metrics = BrokerMetrics()

        assert isinstance(metrics.batch_wait_times, deque)
        assert metrics.batch_wait_times.maxlen == METRICS_HISTORY_MAXLEN

    def test_batch_sizes_evicts_oldest_when_full(self):
        """Test batch_sizes evicts oldest entries when maxlen is exceeded."""
        metrics = BrokerMetrics()

        # Fill beyond maxlen
        for i in range(METRICS_HISTORY_MAXLEN + 10):
            metrics.record_batch_submitted(size=i)

        # Should only keep the most recent METRICS_HISTORY_MAXLEN entries
        assert len(metrics.batch_sizes) == METRICS_HISTORY_MAXLEN
        # Oldest entries (0-9) should be evicted, newest should remain
        assert list(metrics.batch_sizes)[0] == 10
        assert list(metrics.batch_sizes)[-1] == METRICS_HISTORY_MAXLEN + 9

    def test_batch_wait_times_evicts_oldest_when_full(self):
        """Test batch_wait_times evicts oldest entries when maxlen is exceeded."""
        metrics = BrokerMetrics()

        # Fill beyond maxlen
        for i in range(METRICS_HISTORY_MAXLEN + 10):
            metrics.record_batch_completed(wait_seconds=float(i))

        # Should only keep the most recent METRICS_HISTORY_MAXLEN entries
        assert len(metrics.batch_wait_times) == METRICS_HISTORY_MAXLEN
        # Oldest entries should be evicted
        assert list(metrics.batch_wait_times)[0] == 10.0
        assert list(metrics.batch_wait_times)[-1] == float(METRICS_HISTORY_MAXLEN + 9)

    def test_record_request_batched(self):
        """Test recording a batched request."""
        metrics = BrokerMetrics()

        metrics.record_request(batched=True)

        assert metrics.requests_total == 1
        assert metrics.requests_batched == 1
        assert metrics.requests_sync == 0

    def test_record_request_sync(self):
        """Test recording a sync request."""
        metrics = BrokerMetrics()

        metrics.record_request(batched=False)

        assert metrics.requests_total == 1
        assert metrics.requests_batched == 0
        assert metrics.requests_sync == 1

    def test_record_batch_submitted(self):
        """Test recording a batch submission."""
        metrics = BrokerMetrics()

        metrics.record_batch_submitted(size=10)

        assert metrics.batches_submitted == 1
        assert list(metrics.batch_sizes) == [10]

    def test_record_batch_completed(self):
        """Test recording a batch completion."""
        metrics = BrokerMetrics()

        metrics.record_batch_completed(wait_seconds=60.5)

        assert list(metrics.batch_wait_times) == [60.5]

    def test_record_batch_timeout(self):
        """Test recording a batch timeout."""
        metrics = BrokerMetrics()

        metrics.record_batch_timeout()

        assert metrics.requests_batch_timeout == 1

    def test_record_failure(self):
        """Test recording a request failure."""
        metrics = BrokerMetrics()

        metrics.record_failure()

        assert metrics.requests_failed == 1

    def test_record_queue_overflow(self):
        """Test recording queue overflow."""
        metrics = BrokerMetrics()

        metrics.record_queue_overflow()

        assert metrics.queue_overflow_count == 1
        assert metrics.sync_fallback_count == 1

    def test_record_sync_fallback(self):
        """Test recording sync fallback."""
        metrics = BrokerMetrics()

        metrics.record_sync_fallback()

        assert metrics.sync_fallback_count == 1

    def test_to_dict(self):
        """Test exporting metrics to dictionary."""
        metrics = BrokerMetrics()

        # Record some metrics
        metrics.record_request(batched=True)
        metrics.record_request(batched=True)
        metrics.record_request(batched=False)
        metrics.record_batch_submitted(size=2)
        metrics.record_batch_completed(wait_seconds=30.0)

        data = metrics.to_dict()

        assert data["requests_total"] == 3
        assert data["requests_batched"] == 2
        assert data["requests_sync"] == 1
        assert data["batch_rate"] == 2 / 3
        assert data["batches_submitted"] == 1
        assert data["average_batch_size"] == 2.0
        assert data["average_batch_wait_seconds"] == 30.0
        assert "uptime_seconds" in data

    def test_to_dict_with_no_data(self):
        """Test to_dict handles empty metrics."""
        metrics = BrokerMetrics()

        data = metrics.to_dict()

        assert data["requests_total"] == 0
        assert data["batch_rate"] == 0.0  # 0 / max(1, 0) = 0
        assert data["average_batch_size"] == 0.0
        assert data["average_batch_wait_seconds"] == 0.0

    def test_reset(self):
        """Test resetting metrics."""
        metrics = BrokerMetrics()

        # Record some data
        metrics.record_request(batched=True)
        metrics.record_batch_submitted(size=5)
        metrics.record_failure()

        # Reset
        metrics.reset()

        assert metrics.requests_total == 0
        assert metrics.batches_submitted == 0
        assert metrics.requests_failed == 0
        assert len(metrics.batch_sizes) == 0

    def test_thread_safety(self):
        """Test metrics are thread-safe via lock."""
        import threading

        metrics = BrokerMetrics()
        errors = []

        def record_requests():
            try:
                for _ in range(100):
                    metrics.record_request(batched=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert metrics.requests_total == 1000
        assert metrics.requests_batched == 1000
