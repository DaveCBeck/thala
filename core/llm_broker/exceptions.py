"""Custom exceptions for the LLM Broker."""


class BrokerError(Exception):
    """Base exception for all broker errors."""

    pass


class QueueOverflowError(BrokerError):
    """Raised when the queue exceeds max_queue_size and overflow_behavior is 'reject'.

    Attributes:
        queue_size: Current queue size
        max_size: Maximum allowed queue size
    """

    def __init__(self, queue_size: int, max_size: int) -> None:
        self.queue_size = queue_size
        self.max_size = max_size
        super().__init__(f"Queue overflow: {queue_size} requests exceed maximum of {max_size}")


class BatchSubmissionError(BrokerError):
    """Raised when batch submission to Anthropic API fails.

    Attributes:
        request_count: Number of requests in the failed batch
        reason: Reason for failure
    """

    def __init__(self, request_count: int, reason: str) -> None:
        self.request_count = request_count
        self.reason = reason
        super().__init__(f"Failed to submit batch of {request_count} requests: {reason}")


class BrokerNotStartedError(BrokerError):
    """Raised when broker methods are called before start()."""

    def __init__(self) -> None:
        super().__init__("Broker not started. Call broker.start() before making requests.")
