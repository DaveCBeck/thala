"""Prompt caching utilities for cost optimization."""

import logging
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)

CacheTTL = Literal["5m", "1h"]


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


async def invoke_with_cache(
    llm: ChatAnthropic,
    system_prompt: str,
    user_prompt: str,
    cache_system: bool = True,
    cache_ttl: CacheTTL = "5m",
) -> Any:
    """Invoke LLM with prompt caching on system message."""
    messages = create_cached_messages(
        system_content=system_prompt,
        user_content=user_prompt,
        cache_system=cache_system,
        cache_ttl=cache_ttl,
    )

    response = await llm.ainvoke(messages)

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
