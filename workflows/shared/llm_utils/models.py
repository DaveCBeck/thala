"""Model tier definitions and LLM initialization."""

import os
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from core.config import configure_langsmith

configure_langsmith()

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek


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
    SONNET = "claude-sonnet-4-5-20250929"
    SONNET_1M = "claude-sonnet-4-5-20250929"  # Same model as SONNET, requires 1M context beta header
    OPUS = "claude-opus-4-5-20251101"

    # DeepSeek tiers (OpenAI-compatible API)
    DEEPSEEK_V3 = "deepseek-chat"  # V3.2, 128K context, $0.27/$1.10 per MTok
    DEEPSEEK_R1 = "deepseek-reasoner"  # Reasoning model, 128K context, $0.55/$2.19 per MTok


# Beta header for 1M context window (Sonnet 4/4.5 only, Tier 4+)
CONTEXT_1M_BETA = "context-1m-2025-08-07"

# DeepSeek tiers for easy checking
DEEPSEEK_TIERS = {ModelTier.DEEPSEEK_V3, ModelTier.DEEPSEEK_R1}


def is_deepseek_tier(tier: ModelTier) -> bool:
    """Check if a tier is a DeepSeek model."""
    return tier in DEEPSEEK_TIERS


def get_llm(
    tier: ModelTier = ModelTier.SONNET,
    thinking_budget: Optional[int] = None,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """
    Get a configured LLM instance (Claude or DeepSeek).

    Args:
        tier: Model tier selection (HAIKU, SONNET, SONNET_1M, OPUS, DEEPSEEK_V3, DEEPSEEK_R1)
        thinking_budget: Token budget for extended thinking (enables if set).
                        Recommended: 8000-16000 for complex tasks.
                        Supported on Claude models (Sonnet 4.5, Haiku 4.5, Opus 4.5).
                        For DEEPSEEK_R1, thinking is always enabled (explicit mode).
                        Ignored for DEEPSEEK_V3.
        max_tokens: Maximum output tokens (must be > thinking_budget if set)

    Returns:
        BaseChatModel instance configured for the specified tier

    Example:
        # Standard task with Sonnet
        llm = get_llm(ModelTier.SONNET)

        # Large document processing with 1M context
        llm = get_llm(ModelTier.SONNET_1M)

        # Complex analysis with Opus and extended thinking
        llm = get_llm(ModelTier.OPUS, thinking_budget=8000, max_tokens=16000)

        # Cost-effective task with DeepSeek
        llm = get_llm(ModelTier.DEEPSEEK_V3)
    """
    # DeepSeek models use native ChatDeepSeek integration
    if is_deepseek_tier(tier):
        # ChatDeepSeek auto-reads DEEPSEEK_API_KEY and sets LangSmith metadata
        kwargs: dict[str, Any] = {
            "model": tier.value,
            "max_tokens": max_tokens,
            "max_retries": 3,
        }

        if tier == ModelTier.DEEPSEEK_R1:
            # Enable explicit thinking for R1 reasoner model
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            # R1 needs higher max_tokens for reasoning content
            kwargs["max_tokens"] = max(max_tokens, 16384)
            # R1 supports tool calling but NOT tool_choice parameter
            kwargs["disabled_params"] = {"tool_choice": None}
        elif thinking_budget is not None:
            import logging

            logging.getLogger(__name__).warning(
                f"thinking_budget ignored for DeepSeek tier {tier.name} (only R1 supports reasoning)"
            )

        return ChatDeepSeek(**kwargs)

    # Claude models use Anthropic API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    kwargs: dict[str, Any] = {
        "model": tier.value,
        "api_key": api_key,
        "max_tokens": max_tokens,
        "max_retries": 3,  # Handles 429, 500, 502, 503, 529 with exponential backoff
        # LangSmith metadata for automatic cost tracking
        "metadata": {
            "ls_provider": "anthropic",
            "ls_model_name": tier.value,
        },
    }

    # Enable 1M context window beta for SONNET_1M (requires Tier 4+ account)
    if tier == ModelTier.SONNET_1M:
        kwargs["betas"] = [CONTEXT_1M_BETA]

    if thinking_budget is not None:
        if thinking_budget >= max_tokens:
            raise ValueError(
                f"thinking_budget ({thinking_budget}) must be less than max_tokens ({max_tokens})"
            )
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }

    return ChatAnthropic(**kwargs)
