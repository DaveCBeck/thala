"""Configuration for the LLM Broker.

Provides configurable thresholds, timeouts, and behavior settings.
Configuration can be set via environment variables or programmatically.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

from .schemas import UserMode


@dataclass
class BrokerConfig:
    """Configuration for the LLM Broker.

    Attributes:
        enabled: Whether broker routing is enabled (feature flag for rollout)
        default_mode: Default user mode when not specified
        batch_threshold: Submit batch when queue reaches this size
        max_queue_size: Maximum queued requests before overflow protection
        overflow_behavior: What to do when queue overflows ("sync" or "reject")

        initial_wait_hours: First batch response wait timeout
        balanced_retry_hours: Wait timeout for balanced mode retry
        economical_retry_hours: Wait timeout for economical mode retry

        poll_interval_seconds: Seconds between batch status checks
        flush_interval_seconds: Seconds between periodic queue flushes
        max_concurrent_sync: Maximum concurrent synchronous API calls

        queue_dir: Directory for queue persistence files
        enable_metrics: Whether to collect metrics
    """

    # Feature flag - disabled by default for safe rollout
    enabled: bool = False

    # Mode settings
    default_mode: UserMode = UserMode.BALANCED

    # Batch thresholds
    batch_threshold: int = 50
    max_queue_size: int = 100
    overflow_behavior: Literal["sync", "reject"] = "sync"

    # Batch response timeout tiers (for SUBMITTED state)
    initial_wait_hours: float = 1.0
    balanced_retry_hours: float = 3.0
    economical_retry_hours: float = 12.0

    # Polling and concurrency
    poll_interval_seconds: int = 60
    flush_interval_seconds: int = 120
    max_concurrent_sync: int = 5

    # Persistence
    queue_dir: str = field(default_factory=lambda: ".thala/llm_broker")

    # Observability
    enable_metrics: bool = True

    @classmethod
    def from_env(cls) -> "BrokerConfig":
        """Create config from environment variables.

        Environment variables:
            THALA_LLM_BROKER_ENABLED: Enable broker routing (0/1, true/false)
            THALA_LLM_BROKER_MODE: Default mode (fast/balanced/economical)
            THALA_LLM_BROKER_BATCH_THRESHOLD: Batch submission threshold
            THALA_LLM_BROKER_MAX_QUEUE_SIZE: Maximum queue size (also accepts THALA_LLM_BROKER_MAX_QUEUE)
            THALA_LLM_BROKER_OVERFLOW: Overflow behavior (sync/reject)
            THALA_LLM_BROKER_MAX_CONCURRENT_SYNC: Max concurrent synchronous API calls (default: 5)
            THALA_LLM_BROKER_QUEUE_DIR: Queue persistence directory
        """
        enabled_str = os.getenv("THALA_LLM_BROKER_ENABLED", "").lower()
        enabled = enabled_str in ("1", "true", "yes")

        mode_str = os.getenv("THALA_LLM_BROKER_MODE", "balanced").lower()
        mode_map = {
            "fast": UserMode.FAST,
            "balanced": UserMode.BALANCED,
            "economical": UserMode.ECONOMICAL,
        }
        default_mode = mode_map.get(mode_str, UserMode.BALANCED)

        return cls(
            enabled=enabled,
            default_mode=default_mode,
            batch_threshold=int(os.getenv("THALA_LLM_BROKER_BATCH_THRESHOLD", "50")),
            max_queue_size=int(
                os.getenv("THALA_LLM_BROKER_MAX_QUEUE_SIZE", os.getenv("THALA_LLM_BROKER_MAX_QUEUE", "100"))
            ),
            overflow_behavior=("reject" if os.getenv("THALA_LLM_BROKER_OVERFLOW", "sync") == "reject" else "sync"),
            max_concurrent_sync=int(os.getenv("THALA_LLM_BROKER_MAX_CONCURRENT_SYNC", "5")),
            queue_dir=os.getenv("THALA_LLM_BROKER_QUEUE_DIR", ".thala/llm_broker"),
        )

    def get_wait_timeout_hours(self, mode: UserMode, retry_count: int) -> float:
        """Get the appropriate wait timeout based on mode and retry count.

        Args:
            mode: Current user mode
            retry_count: Number of previous retry attempts

        Returns:
            Hours to wait before timeout/retry
        """
        if retry_count == 0:
            return self.initial_wait_hours

        if mode == UserMode.BALANCED:
            # Balanced: 1hr -> 3hr -> sync fallback
            return self.balanced_retry_hours

        if mode == UserMode.ECONOMICAL:
            # Economical: 1hr -> 3hr -> 12hr -> sync fallback
            if retry_count == 1:
                return self.balanced_retry_hours
            return self.economical_retry_hours

        # Fast mode shouldn't batch, but if somehow here, use initial
        return self.initial_wait_hours

    def max_retries_for_mode(self, mode: UserMode) -> int:
        """Get maximum retry attempts before sync fallback.

        Args:
            mode: Current user mode

        Returns:
            Maximum number of retries (0 = no retries, just initial attempt)
        """
        if mode == UserMode.FAST:
            return 0
        if mode == UserMode.BALANCED:
            return 1  # Initial + 1 retry = 2 attempts, 4hr max
        if mode == UserMode.ECONOMICAL:
            return 3  # Initial + 2 retries = 3 attempts
        return 1


# Global config instance
_config: BrokerConfig | None = None


def get_broker_config() -> BrokerConfig:
    """Get or create the global broker configuration."""
    global _config
    if _config is None:
        _config = BrokerConfig.from_env()
    return _config


def set_broker_config(config: BrokerConfig) -> None:
    """Set the global broker configuration (useful for testing)."""
    global _config
    _config = config


def reset_broker_config() -> None:
    """Reset the global broker configuration."""
    global _config
    _config = None


def is_broker_enabled() -> bool:
    """Check if the broker is enabled via feature flag.

    Returns:
        True if THALA_LLM_BROKER_ENABLED=1/true/yes, False otherwise.
    """
    return get_broker_config().enabled
