"""Model tier definitions for LLM selection.

This module provides the foundational ModelTier enum and related utilities
that are used across core/ and workflows/ modules.
"""

from enum import Enum


class ModelTier(Enum):
    """Model tiers for different task complexities.

    Claude tiers:
        HAIKU: Quick tasks, simple text generation
        SONNET: Standard tasks, summarization, metadata extraction (200k context)
        SONNET_1M: Same as SONNET but with 1M token context window for long documents (Tier 4+)
        OPUS: Complex tasks requiring deep analysis (supports extended thinking)

    DeepSeek tiers (10-15x cheaper than Claude, OpenAI-compatible API):
        DEEPSEEK_V3: High-volume simple tasks (classification, filtering, extraction)
        DEEPSEEK_R1: Reasoning tasks (methodology analysis, complex extraction)
    """

    # Claude tiers
    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-6"
    SONNET_1M = "claude-sonnet-4-6"  # Same model as SONNET, requires 1M context beta header
    OPUS = "claude-opus-4-6"

    # DeepSeek tiers (OpenAI-compatible API)
    DEEPSEEK_V3 = "deepseek-chat"  # V3.2, 128K context, $0.27/$1.10 per MTok
    DEEPSEEK_R1 = "deepseek-reasoner"  # Reasoning model, 128K context, $0.55/$2.19 per MTok


# DeepSeek tiers for easy checking
DEEPSEEK_TIERS = {ModelTier.DEEPSEEK_V3, ModelTier.DEEPSEEK_R1}


def is_deepseek_tier(tier: ModelTier) -> bool:
    """Check if a tier is a DeepSeek model.

    Args:
        tier: The model tier to check

    Returns:
        True if the tier is a DeepSeek model, False otherwise
    """
    return tier in DEEPSEEK_TIERS
