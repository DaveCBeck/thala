"""Metrics collection for the LLM Broker.

Provides observability into broker behavior including batch rates,
queue sizes, and response times.
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Maximum entries in batch_sizes and batch_wait_times deques
# 1000 entries provides ~1 day of history at ~40 batches/hour
METRICS_HISTORY_MAXLEN = 1000

logger = logging.getLogger(__name__)


@dataclass
class BrokerMetrics:
    """Metrics for broker observability.

    Thread-safe metrics collection for monitoring broker performance.
    All operations use a lock to ensure consistency.

    Attributes:
        requests_total: Total requests processed
        requests_batched: Requests sent via batch API
        requests_sync: Requests sent via synchronous API
        requests_batch_timeout: Requests that fell back to sync after batch timeout
        requests_failed: Requests that failed
        batches_submitted: Number of batches submitted
        batch_sizes: List of batch sizes for calculating averages
        batch_wait_times: List of wait times (seconds) from submission to response
        queue_overflow_count: Times queue overflow protection triggered
        sync_fallback_count: Times sync fallback was used (overflow or timeout)
    """

    requests_total: int = 0
    requests_batched: int = 0
    requests_sync: int = 0
    requests_batch_timeout: int = 0
    requests_failed: int = 0
    batches_submitted: int = 0
    batch_sizes: deque[int] = field(default_factory=lambda: deque(maxlen=METRICS_HISTORY_MAXLEN))
    batch_wait_times: deque[float] = field(default_factory=lambda: deque(maxlen=METRICS_HISTORY_MAXLEN))
    queue_overflow_count: int = 0
    sync_fallback_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        repr=False,
    )

    def record_request(self, batched: bool) -> None:
        """Record a request being processed.

        Args:
            batched: Whether the request was batched
        """
        with self._lock:
            self.requests_total += 1
            if batched:
                self.requests_batched += 1
            else:
                self.requests_sync += 1

    def record_batch_submitted(self, size: int) -> None:
        """Record a batch being submitted.

        Args:
            size: Number of requests in the batch
        """
        with self._lock:
            self.batches_submitted += 1
            self.batch_sizes.append(size)

    def record_batch_completed(self, wait_seconds: float) -> None:
        """Record a batch completing.

        Args:
            wait_seconds: Time from submission to completion
        """
        with self._lock:
            self.batch_wait_times.append(wait_seconds)

    def record_batch_timeout(self) -> None:
        """Record a batch timeout (request will retry or fall back)."""
        with self._lock:
            self.requests_batch_timeout += 1

    def record_failure(self) -> None:
        """Record a request failure."""
        with self._lock:
            self.requests_failed += 1

    def record_queue_overflow(self) -> None:
        """Record queue overflow protection triggering."""
        with self._lock:
            self.queue_overflow_count += 1
            self.sync_fallback_count += 1

    def record_sync_fallback(self) -> None:
        """Record sync fallback (timeout exhausted retries)."""
        with self._lock:
            self.sync_fallback_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Export metrics for logging/monitoring.

        Returns:
            Dictionary of metric values and computed statistics
        """
        with self._lock:
            total = max(1, self.requests_total)
            batch_count = max(1, len(self.batch_sizes))
            wait_count = max(1, len(self.batch_wait_times))

            return {
                "requests_total": self.requests_total,
                "requests_batched": self.requests_batched,
                "requests_sync": self.requests_sync,
                "requests_failed": self.requests_failed,
                "batch_rate": self.requests_batched / total,
                "batches_submitted": self.batches_submitted,
                "average_batch_size": (sum(self.batch_sizes) / batch_count if self.batch_sizes else 0.0),
                "average_batch_wait_seconds": (
                    sum(self.batch_wait_times) / wait_count if self.batch_wait_times else 0.0
                ),
                "batch_timeout_count": self.requests_batch_timeout,
                "queue_overflow_count": self.queue_overflow_count,
                "sync_fallback_count": self.sync_fallback_count,
                "uptime_seconds": (datetime.now(timezone.utc) - self._started_at).total_seconds(),
            }

    def log_summary(self, level: int = logging.INFO) -> None:
        """Log a summary of current metrics.

        Args:
            level: Logging level to use
        """
        metrics = self.to_dict()
        logger.log(
            level,
            "Broker metrics: "
            f"total={metrics['requests_total']}, "
            f"batched={metrics['requests_batched']} ({metrics['batch_rate']:.1%}), "
            f"sync={metrics['requests_sync']}, "
            f"batches={metrics['batches_submitted']}, "
            f"avg_batch_size={metrics['average_batch_size']:.1f}, "
            f"avg_wait={metrics['average_batch_wait_seconds']:.1f}s",
        )

    def reset(self) -> None:
        """Reset all metrics to initial values."""
        with self._lock:
            self.requests_total = 0
            self.requests_batched = 0
            self.requests_sync = 0
            self.requests_batch_timeout = 0
            self.requests_failed = 0
            self.batches_submitted = 0
            self.batch_sizes.clear()
            self.batch_wait_times.clear()
            self.queue_overflow_count = 0
            self.sync_fallback_count = 0
            self._started_at = datetime.now(timezone.utc)
