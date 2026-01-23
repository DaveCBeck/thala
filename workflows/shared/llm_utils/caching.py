"""Prompt caching utilities for cost optimization.

Supports both:
- Anthropic: Explicit cache_control with ephemeral blocks
- DeepSeek: Automatic prefix-based caching with warmup delay
"""

import asyncio
import logging
import time
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

CacheTTL = Literal["5m", "1h"]

# DeepSeek cache warmup tracking (prefix hash -> timestamp)
_deepseek_cache_warmed: dict[int, float] = {}
DEEPSEEK_CACHE_WARMUP_DELAY = 10.0  # seconds to wait for cache construction


def create_cached_messages(
    system_content: str | list[dict],
    user_content: str,
    cache_system: bool = True,
    cache_ttl: CacheTTL = "5m",
) -> list[dict]:
    """Create messages with cache_control on system content."""
    cache_control = {"type": "ephemeral"}
    if cache_ttl == "1h":
        cache_control["ttl"] = "1h"

    if isinstance(system_content, str):
        system_blocks = [
            {
                "type": "text",
                "text": system_content,
                **({"cache_control": cache_control} if cache_system else {}),
            }
        ]
    else:
        system_blocks = list(system_content)
        if cache_system and system_blocks:
            last_block = dict(system_blocks[-1])
            last_block["cache_control"] = cache_control
            system_blocks[-1] = last_block

    messages = [
        {"role": "system", "content": system_blocks},
        {"role": "user", "content": user_content},
    ]

    return messages


def _is_deepseek_model(llm: BaseChatModel) -> bool:
    """Check if LLM is a DeepSeek model."""
    if isinstance(llm, ChatOpenAI):
        base_url = getattr(llm, "openai_api_base", "") or ""
        return "deepseek" in base_url.lower()
    return False


async def warm_deepseek_cache(system_prompt: str) -> None:
    """Ensure DeepSeek cache is warmed for this prefix.

    DeepSeek's cache is automatic and prefix-based, but requires
    a few seconds to construct after the first request. This function
    tracks which prefixes have been used and waits for cache construction
    on the first call with a new prefix.
    """
    prefix_hash = hash(system_prompt)
    if prefix_hash not in _deepseek_cache_warmed:
        logger.debug(f"DeepSeek cache warmup: waiting {DEEPSEEK_CACHE_WARMUP_DELAY}s for prefix")
        _deepseek_cache_warmed[prefix_hash] = time.time()
        await asyncio.sleep(DEEPSEEK_CACHE_WARMUP_DELAY)


async def invoke_with_cache(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    cache_system: bool = True,
    cache_ttl: CacheTTL = "5m",
) -> Any:
    """Invoke LLM with prompt caching (Anthropic or DeepSeek).

    For Anthropic: Uses explicit cache_control with ephemeral blocks.
    For DeepSeek: Uses automatic prefix-based caching with warmup delay.
    """
    # DeepSeek: automatic prefix caching, needs warmup delay
    if _is_deepseek_model(llm):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Check if cache needs warmup (first call with this prefix)
        prefix_hash = hash(system_prompt)
        if cache_system and prefix_hash not in _deepseek_cache_warmed:
            # First call - make request, then wait for cache construction
            response = await llm.ainvoke(messages)
            _deepseek_cache_warmed[prefix_hash] = time.time()
            logger.debug(f"DeepSeek cache warmup: waiting {DEEPSEEK_CACHE_WARMUP_DELAY}s for prefix")
            await asyncio.sleep(DEEPSEEK_CACHE_WARMUP_DELAY)
            return response

        return await llm.ainvoke(messages)

    # Anthropic: explicit cache_control
    if isinstance(llm, ChatAnthropic) and cache_system:
        messages = create_cached_messages(
            system_content=system_prompt,
            user_content=user_prompt,
            cache_system=cache_system,
            cache_ttl=cache_ttl,
        )
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    response = await llm.ainvoke(messages)

    # Log Anthropic cache usage
    if isinstance(llm, ChatAnthropic):
        usage = getattr(response, "usage_metadata", None)
        if usage:
            details = usage.get("input_token_details", {})
            cache_read = details.get("cache_read", 0)
            cache_creation = details.get("cache_creation", 0)
            if cache_read > 0:
                logger.debug(f"Cache hit: {cache_read} tokens read from cache")
            elif cache_creation > 0:
                logger.debug(f"Cache miss: {cache_creation} tokens written to cache")

    return response
