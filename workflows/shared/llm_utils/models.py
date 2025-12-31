"""Model tier definitions and LLM initialization."""

import os
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from core.config import configure_langsmith

configure_langsmith()

from langchain_anthropic import ChatAnthropic


class ModelTier(Enum):
    """Model tiers for different task complexities.

    HAIKU: Quick tasks, simple text generation
    SONNET: Standard tasks, summarization, metadata extraction
    OPUS: Complex tasks requiring deep analysis (supports extended thinking)
    """
    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-5-20250929"
    OPUS = "claude-opus-4-5-20251101"


def get_llm(
    tier: ModelTier = ModelTier.SONNET,
    thinking_budget: Optional[int] = None,
    max_tokens: int = 4096,
) -> ChatAnthropic:
    """
    Get a configured Anthropic Claude LLM instance.

    Args:
        tier: Model tier selection (HAIKU, SONNET, OPUS)
        thinking_budget: Token budget for extended thinking (enables if set).
                        Recommended: 8000-16000 for complex tasks.
                        Only supported on Sonnet 4.5, Haiku 4.5, and Opus 4.5.
        max_tokens: Maximum output tokens (must be > thinking_budget if set)

    Returns:
        ChatAnthropic instance configured for the specified tier

    Example:
        # Standard task with Sonnet
        llm = get_llm(ModelTier.SONNET)

        # Complex analysis with Opus and extended thinking
        llm = get_llm(ModelTier.OPUS, thinking_budget=8000)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    kwargs: dict[str, Any] = {
        "model": tier.value,
        "api_key": api_key,
        "max_tokens": max_tokens,
    }

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
