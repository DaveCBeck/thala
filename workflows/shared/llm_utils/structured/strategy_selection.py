"""Strategy selection logic for structured output.

Determines which execution strategy to use based on configuration and context.
"""

import logging

from .types import StructuredOutputConfig, StructuredOutputStrategy

logger = logging.getLogger(__name__)


def select_strategy(
    config: StructuredOutputConfig,
    is_batch: bool,
    batch_size: int,
) -> StructuredOutputStrategy:
    """Select optimal strategy based on context.

    Selection rules:
    1. If explicitly specified, use that strategy
    2. If tools provided -> TOOL_AGENT
    3. If thinking_budget set -> LANGCHAIN_STRUCTURED (BATCH_TOOL_CALL incompatible)
    4. If prefer_batch_api=True -> BATCH_TOOL_CALL (for cost savings)
    5. If batch with batch_threshold+ items -> BATCH_TOOL_CALL
    6. Default -> LANGCHAIN_STRUCTURED
    """
    if config.strategy != StructuredOutputStrategy.AUTO:
        logger.debug(f"[DIAG] Strategy explicitly set: {config.strategy}")
        return config.strategy

    if config.tools:
        logger.debug("[DIAG] Strategy: TOOL_AGENT (tools provided)")
        return StructuredOutputStrategy.TOOL_AGENT

    # thinking_budget cannot be used with batch API + tool_choice
    # Fall back to LangChain structured output which handles thinking gracefully
    if config.thinking_budget:
        logger.debug(f"[DIAG] Strategy: LANGCHAIN_STRUCTURED (thinking_budget={config.thinking_budget})")
        return StructuredOutputStrategy.LANGCHAIN_STRUCTURED

    # prefer_batch_api routes ALL requests through batch API for 50% cost savings
    if config.prefer_batch_api:
        logger.debug("[DIAG] Strategy: BATCH_TOOL_CALL (prefer_batch_api=True)")
        return StructuredOutputStrategy.BATCH_TOOL_CALL

    if is_batch and batch_size >= config.batch_threshold:
        logger.debug(f"[DIAG] Strategy: BATCH_TOOL_CALL (batch_size={batch_size} >= threshold={config.batch_threshold})")
        return StructuredOutputStrategy.BATCH_TOOL_CALL

    logger.debug("[DIAG] Strategy: LANGCHAIN_STRUCTURED (default)")
    return StructuredOutputStrategy.LANGCHAIN_STRUCTURED


__all__ = ["select_strategy"]
