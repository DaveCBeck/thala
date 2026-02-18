"""Model tier definitions and LLM initialization.

ModelTier is defined in core/types/model_tier.py and re-exported here
for backwards compatibility. All workflow code should import from here.
"""

import os
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from core.config import configure_langsmith

configure_langsmith()

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

# Re-export ModelTier and related utilities from core/types
# This maintains backwards compatibility for existing imports
from core.types import ModelTier, is_deepseek_tier

# Beta header for 1M context window (Sonnet 4/4.5 only, Tier 4+)
CONTEXT_1M_BETA = "context-1m-2025-08-07"


def get_llm(
    tier: ModelTier = ModelTier.SONNET,
    effort: Optional[str] = None,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """
    Get a configured LLM instance (Claude or DeepSeek).

    DEPRECATED: Use invoke() instead for all LLM calls. This function is
    retained for internal use by invoke() and structured output executors.
    New code should always use invoke() from workflows.shared.llm_utils.

    Args:
        tier: Model tier selection (HAIKU, SONNET, SONNET_1M, OPUS, DEEPSEEK_V3, DEEPSEEK_R1)
        effort: Adaptive thinking effort level ("low", "medium", "high", "max").
                For DEEPSEEK_R1, thinking is always enabled (explicit mode).
                Ignored for DEEPSEEK_V3.
        max_tokens: Maximum output tokens

    Returns:
        BaseChatModel instance configured for the specified tier

    Example:
        # PREFERRED: Use invoke() for all LLM calls
        response = await invoke(tier=ModelTier.SONNET, system="...", user="...")

        # DEPRECATED: Direct get_llm() usage
        llm = get_llm(ModelTier.SONNET)
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
        elif effort is not None:
            import logging

            logging.getLogger(__name__).warning(
                f"effort ignored for DeepSeek tier {tier.name} (only R1 supports reasoning)"
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

    if effort is not None:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["effort"] = effort

    return ChatAnthropic(**kwargs)
