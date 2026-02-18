"""Configuration for the unified invoke() function.

This module provides the InvokeConfig dataclass that controls caching,
batching, extended thinking, and other LLM invocation parameters.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import BaseTool

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
        effort: Adaptive thinking effort level ("low", "medium", "high", "max").
            Anthropic only. Higher effort = more thinking tokens.
        tools: Tool definitions for tool use. List of tool dicts or LangChain tools.
        tool_choice: Tool choice configuration (e.g., {"type": "auto"}).
        metadata: Additional metadata for tracking and observability.
        max_tokens: Maximum output tokens (default: 4096).

        # Structured output options (used when schema= is provided):
        max_retries: Maximum retry attempts on extraction failure (default: 2).
        retry_backoff: Backoff multiplier between retries (default: 2.0).
        enable_context_fallback: Auto-upgrade to SONNET_1M on context limit errors.
        use_json_schema_method: Use method="json_schema" for stricter validation.
        max_tool_calls: Maximum tool calls for TOOL_AGENT strategy (default: 12).
        max_tool_result_chars: Maximum chars from tool results (default: 100000).

    Example:
        # Simple cached call
        config = InvokeConfig()

        # With batching
        config = InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE)

        # Adaptive thinking
        config = InvokeConfig(effort="high", cache=False)

        # With tools
        config = InvokeConfig(tools=[my_tool], tool_choice={"type": "auto"})

        # Structured output with tools (TOOL_AGENT strategy)
        config = InvokeConfig(tools=[search_tool], max_tool_calls=10)
    """

    # Caching
    cache: bool = True
    cache_ttl: Literal["5m", "1h"] = "5m"

    # Batching
    batch_policy: "BatchPolicy | None" = None

    # Adaptive thinking (Anthropic models only)
    effort: str | None = None

    # Tools (for both text and structured output)
    tools: list[BaseTool] | list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tokens
    max_tokens: int = 4096

    # Concurrency control for direct invocation
    max_concurrent: int = 10

    # Structured output options
    max_retries: int = 2
    retry_backoff: float = 2.0
    enable_context_fallback: bool = True
    use_json_schema_method: bool = False
    max_tool_calls: int = 12
    max_tool_result_chars: int = 100000

    def __post_init__(self) -> None:
        """Validate constraint combinations.

        Note: No constraint validation needed — adaptive thinking is compatible
        with prompt caching on Anthropic.
        """
        pass
