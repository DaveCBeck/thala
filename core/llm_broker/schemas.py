"""Data models for the LLM Broker.

This module defines the core data structures for request routing,
state tracking, and batch policy configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid


class BatchPolicy(Enum):
    """Call-site batch policy declaration.

    Determines how a request should be routed based on user mode.
    """

    FORCE_BATCH = "force_batch"  # Always batch regardless of mode (use sparingly)
    PREFER_BALANCE = "prefer_balance"  # Batch in Balanced or Economical mode
    PREFER_SPEED = "prefer_speed"  # Batch only in Economical mode
    REQUIRE_SYNC = "sync"  # Always use synchronous API, never batch


class UserMode(Enum):
    """User-configurable processing mode.

    Controls the tradeoff between speed and cost optimization.
    """

    FAST = "fast"  # No batching; all calls use synchronous API
    BALANCED = "balanced"  # Batch calls marked PREFER_BALANCE or below
    ECONOMICAL = "economical"  # Aggressive batching; batch all eligible calls


class RequestState(Enum):
    """Lifecycle state of a queued request."""

    QUEUED = "queued"  # In queue, waiting for batch submission
    SUBMITTED = "submitted"  # Batch submitted to API, awaiting response
    COMPLETED = "completed"  # Response received successfully
    FAILED = "failed"  # Request failed (API error, validation, etc.)


@dataclass
class LLMRequest:
    """A request to be processed by the broker.

    Attributes:
        request_id: Unique identifier for this request
        prompt: The user message/prompt to send
        model: Model identifier string (e.g., "claude-sonnet-4-5-20250929")
        policy: Batch policy for this request
        state: Current lifecycle state
        max_tokens: Maximum output tokens
        system: Optional system prompt
        thinking_budget: Optional token budget for extended thinking
        tools: Optional tool definitions for structured output
        tool_choice: Optional tool choice configuration
        created_at: When the request was created
        submitted_at: When the batch was submitted to API
        batch_id: Anthropic batch ID (set after submission)
        retry_count: Number of retry attempts for this request
        metadata: Additional metadata for tracking
    """

    request_id: str
    prompt: str
    model: str
    policy: BatchPolicy = BatchPolicy.PREFER_SPEED
    state: RequestState = RequestState.QUEUED
    max_tokens: int = 4096
    system: str | None = None
    thinking_budget: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: datetime | None = None
    batch_id: str | None = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        prompt: str,
        model: str,
        policy: BatchPolicy = BatchPolicy.PREFER_SPEED,
        **kwargs: Any,
    ) -> "LLMRequest":
        """Factory method to create a new request with auto-generated ID."""
        return cls(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            model=model,
            policy=policy,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize request for persistence."""
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "model": self.model,
            "policy": self.policy.value,
            "state": self.state.value,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "thinking_budget": self.thinking_budget,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "batch_id": self.batch_id,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMRequest":
        """Deserialize from dictionary with validation."""
        required_fields = ["request_id", "prompt", "model", "policy", "state"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing required fields in queue data: {missing}")

        try:
            return cls(
                request_id=data["request_id"],
                prompt=data["prompt"],
                model=data["model"],
                policy=BatchPolicy(data["policy"]),
                state=RequestState(data["state"]),
                max_tokens=data.get("max_tokens", 4096),
                system=data.get("system"),
                thinking_budget=data.get("thinking_budget"),
                tools=data.get("tools"),
                tool_choice=data.get("tool_choice"),
                created_at=datetime.fromisoformat(data["created_at"]),
                submitted_at=(datetime.fromisoformat(data["submitted_at"]) if data.get("submitted_at") else None),
                batch_id=data.get("batch_id"),
                retry_count=data.get("retry_count", 0),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid queue data: {e}") from e


@dataclass
class LLMResponse:
    """Response from an LLM request.

    Attributes:
        request_id: ID of the original request
        content: Response content (text or tool use result)
        success: Whether the request succeeded
        error: Error message if failed
        usage: Token usage statistics
        model: Model that processed the request
        stop_reason: Why generation stopped
        thinking: Extended thinking content if available
    """

    request_id: str
    content: Any
    success: bool = True
    error: str | None = None
    usage: dict[str, int] | None = None
    model: str | None = None
    stop_reason: str | None = None
    thinking: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize response for logging/storage."""
        return {
            "request_id": self.request_id,
            "content": self.content,
            "success": self.success,
            "error": self.error,
            "usage": self.usage,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "thinking": self.thinking,
        }
