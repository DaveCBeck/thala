"""Prompt caching utilities for cost optimization.

Supports both:
- Anthropic: Explicit cache_control with ephemeral blocks
- DeepSeek: Automatic prefix-based caching with warmup delay

Note: For LLM invocation, use invoke() from workflows.shared.llm_utils instead.
This module provides caching utilities used internally by invoke().
"""

import asyncio
import logging
import time
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

CacheTTL = Literal["5m", "1h"]


class BrokerResponseWrapper:
    """Wraps broker LLMResponse to be compatible with LangChain response format.

    This allows code using extract_response_content() and similar utilities
    to work seamlessly with responses from either LangChain models or the broker.
    """

    def __init__(self, content: Any, usage: Optional[dict] = None):
        self.content = content
        self.usage_metadata = usage


# DeepSeek cache warmup tracking (prefix hash -> timestamp)
_deepseek_cache_warmed: dict[int, float] = {}
DEEPSEEK_CACHE_WARMUP_DELAY = 10.0  # seconds to wait for cache construction


def create_cached_messages(
    system_content: str | list[dict],
    user_content: str,
    cache_system: bool = True,
    cache_ttl: CacheTTL = "5m",
) -> list[dict]:
    """Create messages with cache_control on system content.

    Args:
        system_content: System message as string or list of content blocks
        user_content: User message
        cache_system: Whether to apply cache_control to system (default: True)
        cache_ttl: "5m" (default, ephemeral) or "1h" (longer retention)

    Returns:
        List of messages with cache_control applied to system content
    """
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


async def warm_deepseek_cache(system_prompt: str) -> None:
    """Ensure DeepSeek cache is warmed for this prefix.

    DeepSeek's cache is automatic and prefix-based, but requires
    a few seconds to construct after the first request. This function
    tracks which prefixes have been used and waits for cache construction
    on the first call with a new prefix.

    Args:
        system_prompt: The system prompt to warm cache for
    """
    prefix_hash = hash(system_prompt)
    if prefix_hash not in _deepseek_cache_warmed:
        logger.debug(f"DeepSeek cache warmup: waiting {DEEPSEEK_CACHE_WARMUP_DELAY}s for prefix")
        _deepseek_cache_warmed[prefix_hash] = time.time()
        await asyncio.sleep(DEEPSEEK_CACHE_WARMUP_DELAY)
