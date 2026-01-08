"""Strategy selection logic for structured output.

Determines which execution strategy to use based on configuration and context.
"""

from .types import StructuredOutputConfig, StructuredOutputStrategy


def select_strategy(
    config: StructuredOutputConfig,
    is_batch: bool,
    batch_size: int,
) -> StructuredOutputStrategy:
    """Select optimal strategy based on context.

    Selection rules:
    1. If explicitly specified, use that strategy
    2. If tools provided -> TOOL_AGENT
    3. If prefer_batch_api=True -> BATCH_TOOL_CALL (for cost savings)
    4. If batch with batch_threshold+ items -> BATCH_TOOL_CALL
    5. Default -> LANGCHAIN_STRUCTURED
    """
    if config.strategy != StructuredOutputStrategy.AUTO:
        return config.strategy

    if config.tools:
        return StructuredOutputStrategy.TOOL_AGENT

    # prefer_batch_api routes ALL requests through batch API for 50% cost savings
    if config.prefer_batch_api:
        return StructuredOutputStrategy.BATCH_TOOL_CALL

    if is_batch and batch_size >= config.batch_threshold:
        return StructuredOutputStrategy.BATCH_TOOL_CALL

    return StructuredOutputStrategy.LANGCHAIN_STRUCTURED


__all__ = ["select_strategy"]
