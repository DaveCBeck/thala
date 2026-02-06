"""Configuration for the unified invoke() function.

This module provides the InvokeConfig dataclass that controls caching,
batching, extended thinking, and other LLM invocation parameters.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from core.llm_broker import BatchPolicy


@dataclass
class InvokeConfig:
    """Configuration for invoke() calls.

    Attributes:
        cache: Enable prompt caching (default: True). For Anthropic, uses
            ephemeral cache_control blocks. For DeepSeek, uses automatic
            prefix-based caching.
        cache_ttl: Cache time-to-live. "5m" (default) or "1h" for longer retention.
        batch_policy: When set and broker is enabled, routes requests through
            the central LLM broker for cost optimization. Use BatchPolicy enum
            values (FORCE_BATCH, PREFER_BALANCE, PREFER_SPEED, REQUIRE_SYNC).
        thinking_budget: Token budget for extended thinking (Anthropic only).
            Recommended: 8000-16000 for complex tasks. Cannot be used with cache=True.
        tools: Tool definitions for tool use. List of tool dicts or LangChain tools.
        tool_choice: Tool choice configuration (e.g., {"type": "auto"}).
        metadata: Additional metadata for tracking and observability.
        max_tokens: Maximum output tokens (default: 4096).

    Example:
        # Simple cached call
        config = InvokeConfig()

        # With batching
        config = InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE)

        # Extended thinking (cache must be disabled)
        config = InvokeConfig(thinking_budget=8000, cache=False)

        # With tools
        config = InvokeConfig(tools=[my_tool], tool_choice={"type": "auto"})
    """

    # Caching
    cache: bool = True
    cache_ttl: Literal["5m", "1h"] = "5m"

    # Batching
    batch_policy: "BatchPolicy | None" = None

    # Extended thinking (Anthropic models only)
    thinking_budget: int | None = None

    # Tools
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tokens
    max_tokens: int = 4096

    # Concurrency control for direct invocation
    max_concurrent: int = 10

    def __post_init__(self) -> None:
        """Validate constraint combinations.

        Note: Cache + thinking_budget validation is deferred to invoke()
        where we know the model tier. DeepSeek R1 allows this combination
        since it has automatic prefix caching independent of thinking.
        """
        pass
