"""Type definitions for structured output extraction.

Contains enums, dataclasses, and result types used across the structured output module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Generic, Optional, Type, TypeVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ..caching import CacheTTL
from ..models import ModelTier

T = TypeVar("T", bound=BaseModel)


class StructuredOutputStrategy(Enum):
    """Strategy for obtaining structured output."""

    LANGCHAIN_STRUCTURED = auto()  # .with_structured_output()
    BATCH_TOOL_CALL = auto()  # Anthropic Batch API with tools
    TOOL_AGENT = auto()  # Multi-turn agent with tools
    JSON_PROMPTING = auto()  # Fallback JSON extraction
    AUTO = auto()  # Auto-select based on context


@dataclass
class StructuredRequest:
    """A single request in a batch.

    Args:
        id: Unique identifier for this request (used to retrieve results)
        user_prompt: The user message content
        system_prompt: Optional per-request system prompt (overrides default)
    """

    id: str
    user_prompt: str
    system_prompt: Optional[str] = None


def _get_prefer_batch_default() -> bool:
    """Get default for prefer_batch_api from environment variable."""
    return os.getenv("THALA_PREFER_BATCH_API", "").lower() in ("true", "1", "yes")


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output extraction.

    Args:
        tier: Model tier (HAIKU, SONNET, SONNET_1M, OPUS)
        max_tokens: Maximum output tokens
        thinking_budget: Token budget for extended thinking (Opus)

        strategy: Force a specific strategy (default: AUTO)
        use_json_schema_method: Use method="json_schema" for stricter validation
        prefer_batch_api: Route ALL requests through batch API for 50% cost savings.
            When True, even single requests use the batch API (higher latency but cheaper).
            Defaults to THALA_PREFER_BATCH_API environment variable.
        batch_threshold: Minimum items to trigger batch API when prefer_batch_api=False (default: 5)

        max_retries: Maximum retry attempts on failure
        retry_backoff: Backoff multiplier between retries

        enable_prompt_cache: Enable prompt caching (90% savings on cache hits)
        cache_ttl: Prompt cache TTL ("5m" or "1h")

        tools: LangChain tools for TOOL_AGENT strategy
        max_tool_calls: Maximum tool calls for agent
        max_tool_result_chars: Maximum chars from tool results
    """

    # Model configuration
    tier: ModelTier = ModelTier.SONNET
    max_tokens: int = 4096
    thinking_budget: Optional[int] = None

    # Strategy selection
    strategy: StructuredOutputStrategy = StructuredOutputStrategy.AUTO
    use_json_schema_method: bool = False
    prefer_batch_api: bool = field(default_factory=_get_prefer_batch_default)
    batch_threshold: int = 5  # Use batch API for 5+ items (when prefer_batch_api=False)

    # Retry behavior
    max_retries: int = 2
    retry_backoff: float = 2.0

    # Caching
    enable_prompt_cache: bool = True
    cache_ttl: CacheTTL = "5m"

    # Tool agent specific
    tools: list[BaseTool] = field(default_factory=list)
    max_tool_calls: int = 12
    max_tool_result_chars: int = 100000


@dataclass
class StructuredOutputResult(Generic[T]):
    """Result from structured output extraction.

    Attributes:
        success: Whether extraction succeeded
        value: The extracted value (if success=True)
        error: Error message (if success=False)
        strategy_used: Which strategy was used
        usage: Token usage metadata
        thinking: Extended thinking content (if enabled)
    """

    success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    strategy_used: Optional[StructuredOutputStrategy] = None
    usage: Optional[dict] = None
    thinking: Optional[str] = None

    @classmethod
    def ok(
        cls,
        value: T,
        strategy: StructuredOutputStrategy,
        usage: Optional[dict] = None,
        thinking: Optional[str] = None,
    ) -> "StructuredOutputResult[T]":
        """Create a successful result."""
        return cls(
            success=True,
            value=value,
            strategy_used=strategy,
            usage=usage,
            thinking=thinking,
        )

    @classmethod
    def err(cls, error: str) -> "StructuredOutputResult[T]":
        """Create a failed result."""
        return cls(success=False, error=error)


@dataclass
class BatchResult(Generic[T]):
    """Results from batch structured output.

    Attributes:
        results: Mapping of request ID to result
        total_items: Total items processed
        successful_items: Count of successful extractions
        failed_items: Count of failed extractions
        strategy_used: Which strategy was used
    """

    results: dict[str, StructuredOutputResult[T]]
    total_items: int
    successful_items: int
    failed_items: int
    strategy_used: StructuredOutputStrategy


class StructuredOutputError(Exception):
    """Raised when structured output extraction fails after retries."""

    def __init__(
        self,
        message: str,
        schema: Type[BaseModel],
        strategy: StructuredOutputStrategy,
        attempts: int,
        last_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.schema = schema
        self.strategy = strategy
        self.attempts = attempts
        self.last_error = last_error


__all__ = [
    "StructuredOutputStrategy",
    "StructuredRequest",
    "StructuredOutputConfig",
    "StructuredOutputResult",
    "BatchResult",
    "StructuredOutputError",
    "_get_prefer_batch_default",
]
