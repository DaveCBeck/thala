"""Strategy selection logic for structured output.

Determines which execution strategy to use based on configuration and context.
"""

import logging

from ..models import is_deepseek_tier
from .types import StructuredOutputConfig, StructuredOutputStrategy

logger = logging.getLogger(__name__)


def select_strategy(
    config: StructuredOutputConfig,
    is_batch: bool,
) -> StructuredOutputStrategy:
    """Select optimal strategy based on context.

    Selection rules:
    1. If explicitly specified, use that strategy
    2. If tools provided -> TOOL_AGENT
    3. If DeepSeek tier -> LANGCHAIN_STRUCTURED (no batch API available)
    4. If thinking_budget set -> LANGCHAIN_STRUCTURED (BATCH_TOOL_CALL incompatible)
    5. Default -> LANGCHAIN_STRUCTURED

    Note: The is_batch parameter is retained for logging context and future flexibility.
    Note: Batch API routing is now handled via batch_policy in StructuredOutputConfig.
    """
    if config.strategy != StructuredOutputStrategy.AUTO:
        logger.debug(f"[DIAG] Strategy explicitly set: {config.strategy}")
        return config.strategy

    if config.tools:
        logger.debug("[DIAG] Strategy: TOOL_AGENT (tools provided)")
        return StructuredOutputStrategy.TOOL_AGENT

    # DeepSeek doesn't have a batch API - always use synchronous calls
    if is_deepseek_tier(config.tier):
        logger.debug(f"[DIAG] Strategy: LANGCHAIN_STRUCTURED (DeepSeek tier {config.tier.name}, no batch API)")
        return StructuredOutputStrategy.LANGCHAIN_STRUCTURED

    # thinking_budget cannot be used with batch API + tool_choice
    # Fall back to LangChain structured output which handles thinking gracefully
    if config.thinking_budget:
        logger.debug(f"[DIAG] Strategy: LANGCHAIN_STRUCTURED (thinking_budget={config.thinking_budget})")
        return StructuredOutputStrategy.LANGCHAIN_STRUCTURED

    logger.debug("[DIAG] Strategy: LANGCHAIN_STRUCTURED (default)")
    return StructuredOutputStrategy.LANGCHAIN_STRUCTURED


__all__ = ["select_strategy"]
