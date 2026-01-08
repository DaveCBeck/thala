"""Prompt caching utilities for cost optimization."""

import asyncio
import json
import logging
from typing import Any, Optional, Literal

from langchain_anthropic import ChatAnthropic

from .models import ModelTier, get_llm
from ..retry_utils import with_retry
from .response_parsing import extract_json_from_response, extract_response_content

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
            logger.debug(f"Cache HIT: {cache_read} tokens read from cache")
        elif cache_creation > 0:
            logger.debug(f"Cache MISS: {cache_creation} tokens written to cache")

    return response


async def summarize_text_cached(
    text: str,
    target_words: int = 100,
    context: Optional[str] = None,
    tier: ModelTier = ModelTier.SONNET,
) -> str:
    """Summarize text using Claude with prompt caching."""
    llm = get_llm(tier=tier)

    system_prompt = f"""You are a skilled summarizer. Create concise summaries that capture the essential information.

Target length: approximately {target_words} words.
{f"Context: {context}" if context else ""}

Guidelines:
- Focus on the main thesis, key arguments, and conclusions
- Preserve critical details and nuance
- Write in clear, professional prose"""

    user_prompt = f"Summarize the following text:\n\n{text}"

    async def _invoke():
        response = await invoke_with_cache(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return extract_response_content(response)

    return await with_retry(_invoke)


async def extract_json_cached(
    text: str,
    system_instructions: str,
    schema_hint: Optional[str] = None,
    tier: ModelTier = ModelTier.SONNET,
) -> dict[str, Any]:
    """Extract structured JSON from text using Claude with prompt caching.

    DEPRECATED: Consider using get_structured_output(enable_prompt_cache=True)
    instead for type-safe extraction with automatic caching.
    """
    llm = get_llm(tier=tier)

    system_prompt = system_instructions
    if schema_hint:
        system_prompt += f"\n\nExpected schema:\n{schema_hint}"
    system_prompt += "\n\nRespond with ONLY valid JSON, no other text."

    user_prompt = f"Extract from this text:\n\n{text}"

    async def _invoke():
        response = await invoke_with_cache(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        content = extract_response_content(response)
        return extract_json_from_response(content)

    try:
        return await with_retry(_invoke, retry_on=json.JSONDecodeError)
    except RuntimeError as e:
        if "JSONDecodeError" in str(e.__cause__):
            raise json.JSONDecodeError(
                f"Failed to parse JSON after retries: {str(e.__cause__)}",
                "",
                0,
            ) from e
        raise
