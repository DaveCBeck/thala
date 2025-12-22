"""LLM utilities for document processing workflows.

This module provides Anthropic Claude model integration with:
- Tiered model selection (Haiku/Sonnet/Opus)
- Extended thinking support for complex reasoning tasks
- Prompt caching for cost optimization (90% savings on cache hits)
- Both synchronous and batch processing modes

Prompt Caching:
    Cache reads cost 10% of base input token price. To use caching:
    1. Use invoke_with_cache() for simple cached calls
    2. Use create_cached_messages() to build messages with cache_control
    3. Structure prompts with static content first, dynamic content last
"""

import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Optional, Literal

from dotenv import load_dotenv

load_dotenv()

from core.config import configure_langsmith

configure_langsmith()

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)


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


async def summarize_text(
    text: str,
    target_words: int = 100,
    context: Optional[str] = None,
    tier: ModelTier = ModelTier.SONNET,
) -> str:
    """
    Summarize text using Claude with retry logic.

    Args:
        text: Text to summarize
        target_words: Target word count for summary
        context: Optional context about the document
        tier: Model tier to use (default: SONNET)

    Returns:
        Summary text
    """
    llm = get_llm(tier=tier)

    prompt = f"Summarize the following text in approximately {target_words} words."
    if context:
        prompt += f"\n\nContext: {context}"
    prompt += f"\n\nText:\n{text}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to summarize after {max_retries} attempts") from e


async def extract_json(
    text: str,
    prompt: str,
    schema_hint: Optional[str] = None,
    tier: ModelTier = ModelTier.SONNET,
) -> dict[str, Any]:
    """
    Extract structured JSON from text using Claude.

    Args:
        text: Text to extract from
        prompt: Instructions for extraction
        schema_hint: Optional JSON schema hint for expected output
        tier: Model tier to use (default: SONNET)

    Returns:
        Extracted data as dict

    Raises:
        RuntimeError: If extraction fails after retries
        json.JSONDecodeError: If LLM output is not valid JSON
    """
    llm = get_llm(tier=tier)

    full_prompt = prompt
    if schema_hint:
        full_prompt += f"\n\nExpected schema:\n{schema_hint}"
    full_prompt += "\n\nRespond with ONLY valid JSON, no other text."
    full_prompt += f"\n\nText:\n{text}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke([HumanMessage(content=full_prompt)])
            content = response.content.strip()

            # Try to extract JSON from markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                # Remove first line (```json or ```) and last line (```)
                content = "\n".join(lines[1:-1])

            return json.loads(content)
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                raise json.JSONDecodeError(
                    f"Failed to parse JSON after {max_retries} attempts: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to extract JSON after {max_retries} attempts") from e


async def analyze_with_thinking(
    text: str,
    prompt: str,
    thinking_budget: int = 8000,
    tier: ModelTier = ModelTier.OPUS,
) -> tuple[str, Optional[str]]:
    """
    Perform complex analysis using extended thinking.

    Extended thinking allows Claude to reason step-by-step before
    providing a final answer, improving quality for complex tasks.

    Args:
        text: Text to analyze
        prompt: Analysis instructions
        thinking_budget: Token budget for reasoning (default: 8000)
        tier: Model tier (default: OPUS for complex analysis)

    Returns:
        Tuple of (final_response, thinking_summary)
        thinking_summary contains Claude's reasoning process (summarized)
    """
    llm = get_llm(tier=tier, thinking_budget=thinking_budget, max_tokens=thinking_budget + 4096)

    full_prompt = f"{prompt}\n\nText:\n{text}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke([HumanMessage(content=full_prompt)])

            # Extract thinking and text content from response
            thinking_content = None
            text_content = ""

            # Response content may be a list of content blocks or a string
            if isinstance(response.content, list):
                for block in response.content:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            thinking_content = block.get("thinking", "")
                        elif block.get("type") == "text":
                            text_content = block.get("text", "")
                    elif hasattr(block, "type"):
                        if block.type == "thinking":
                            thinking_content = getattr(block, "thinking", "")
                        elif block.type == "text":
                            text_content = getattr(block, "text", "")
            else:
                text_content = response.content

            return text_content, thinking_content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to analyze after {max_retries} attempts") from e


# =============================================================================
# Prompt Caching Utilities
# =============================================================================

CacheTTL = Literal["5m", "1h"]


def create_cached_messages(
    system_content: str | list[dict],
    user_content: str,
    cache_system: bool = True,
    cache_ttl: CacheTTL = "5m",
) -> list[dict]:
    """
    Create messages with cache_control on system content.

    Prompt caching reduces input token costs by 90% on cache hits.
    The cache is automatically refreshed for 5 minutes (or 1 hour with extended TTL).

    Args:
        system_content: Static system prompt (string or list of content blocks)
        user_content: Dynamic user message content
        cache_system: Whether to apply cache_control to system content (default: True)
        cache_ttl: Cache lifetime - "5m" (default, free refresh) or "1h" (2x write cost)

    Returns:
        List of message dicts ready for llm.invoke()

    Example:
        messages = create_cached_messages(
            system_content="You are a research assistant...",
            user_content=f"Analyze: {dynamic_text}",
        )
        response = await llm.ainvoke(messages)
    """
    cache_control = {"type": "ephemeral"}
    if cache_ttl == "1h":
        cache_control["ttl"] = "1h"

    # Build system message content blocks
    if isinstance(system_content, str):
        system_blocks = [
            {
                "type": "text",
                "text": system_content,
                **({"cache_control": cache_control} if cache_system else {}),
            }
        ]
    else:
        # Already a list of content blocks - add cache_control to last block
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
    """
    Invoke LLM with prompt caching on system message.

    This is a convenience wrapper that:
    1. Creates properly structured messages with cache_control
    2. Invokes the LLM
    3. Logs cache hit/miss statistics

    Args:
        llm: ChatAnthropic instance
        system_prompt: Static system instructions (will be cached)
        user_prompt: Dynamic user content
        cache_system: Whether to cache the system prompt (default: True)
        cache_ttl: Cache lifetime - "5m" or "1h"

    Returns:
        LLM response

    Example:
        llm = get_llm(ModelTier.SONNET)
        response = await invoke_with_cache(
            llm,
            system_prompt=METADATA_EXTRACTION_PROMPT,  # Cached
            user_prompt=f"Extract from: {document_text}",  # Dynamic
        )
    """
    messages = create_cached_messages(
        system_content=system_prompt,
        user_content=user_prompt,
        cache_system=cache_system,
        cache_ttl=cache_ttl,
    )

    response = await llm.ainvoke(messages)

    # Log cache statistics if available
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
    """
    Summarize text using Claude with prompt caching.

    The system instructions are cached, so repeated calls with different
    texts will benefit from ~90% input token cost reduction.

    Args:
        text: Text to summarize
        target_words: Target word count for summary
        context: Optional context about the document
        tier: Model tier to use (default: SONNET)

    Returns:
        Summary text
    """
    llm = get_llm(tier=tier)

    # Static system prompt (cached)
    system_prompt = f"""You are a skilled summarizer. Create concise summaries that capture the essential information.

Target length: approximately {target_words} words.
{f"Context: {context}" if context else ""}

Guidelines:
- Focus on the main thesis, key arguments, and conclusions
- Preserve critical details and nuance
- Write in clear, professional prose"""

    # Dynamic user content
    user_prompt = f"Summarize the following text:\n\n{text}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await invoke_with_cache(
                llm,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            return response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to summarize after {max_retries} attempts") from e


async def extract_json_cached(
    text: str,
    system_instructions: str,
    schema_hint: Optional[str] = None,
    tier: ModelTier = ModelTier.SONNET,
) -> dict[str, Any]:
    """
    Extract structured JSON from text using Claude with prompt caching.

    The system instructions and schema are cached, so repeated extractions
    (e.g., processing multiple documents) benefit from ~90% cost reduction.

    Args:
        text: Text to extract from
        system_instructions: Instructions for extraction (will be cached)
        schema_hint: Optional JSON schema hint for expected output
        tier: Model tier to use (default: SONNET)

    Returns:
        Extracted data as dict
    """
    llm = get_llm(tier=tier)

    # Build cached system prompt
    system_prompt = system_instructions
    if schema_hint:
        system_prompt += f"\n\nExpected schema:\n{schema_hint}"
    system_prompt += "\n\nRespond with ONLY valid JSON, no other text."

    # Dynamic user content
    user_prompt = f"Extract from this text:\n\n{text}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await invoke_with_cache(
                llm,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
            content = content.strip()

            # Try to extract JSON from markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            return json.loads(content)
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                raise json.JSONDecodeError(
                    f"Failed to parse JSON after {max_retries} attempts: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to extract JSON after {max_retries} attempts") from e
