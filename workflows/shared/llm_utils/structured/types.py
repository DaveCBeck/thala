"""Type definitions for structured output extraction.

Contains enums, dataclasses, and result types used across the structured output module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Generic, Optional, Type, TypeVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ..caching import CacheTTL
from ..models import ModelTier

if TYPE_CHECKING:
    from core.llm_broker import BatchPolicy

T = TypeVar("T", bound=BaseModel)


class StructuredOutputStrategy(Enum):
    """Strategy for obtaining structured output."""

    LANGCHAIN_STRUCTURED = auto()  # .with_structured_output()
    BATCH_TOOL_CALL = auto()  # Anthropic Batch API with tools
    TOOL_AGENT = auto()  # Multi-turn agent with tools
    JSON_PROMPTING = auto()  # Fallback JSON extraction
    AUTO = auto()  # Auto-select based on context


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output extraction.

    Args:
        tier: Model tier (HAIKU, SONNET, SONNET_1M, OPUS, DEEPSEEK_V3, DEEPSEEK_R1)
        max_tokens: Maximum output tokens
        effort: Adaptive thinking effort level ("low", "medium", "high", "max")

        strategy: Force a specific strategy (default: AUTO)
        use_json_schema_method: Use method="json_schema" for stricter validation

        max_retries: Maximum retry attempts on failure
        retry_backoff: Backoff multiplier between retries

        enable_prompt_cache: Enable prompt caching (90% savings on cache hits)
        cache_ttl: Prompt cache TTL ("5m" or "1h")

        tools: LangChain tools for TOOL_AGENT strategy
        max_tool_calls: Maximum tool calls for agent
        max_tool_result_chars: Maximum chars from tool results

        batch_policy: Routes through central LLM broker with specified policy
    """

    # Model configuration
    tier: ModelTier = ModelTier.SONNET
    max_tokens: int = 4096
    effort: Optional[str] = None

    # Strategy selection
    strategy: StructuredOutputStrategy = StructuredOutputStrategy.AUTO
    use_json_schema_method: bool = False

    # Broker integration - routes through central LLM broker with specified policy
    batch_policy: Optional["BatchPolicy"] = None

    # Retry behavior
    max_retries: int = 2
    retry_backoff: float = 2.0
    enable_context_fallback: bool = True  # Auto-upgrade to SONNET_1M on context limit errors

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
    "StructuredOutputConfig",
    "StructuredOutputResult",
    "StructuredOutputError",
]
