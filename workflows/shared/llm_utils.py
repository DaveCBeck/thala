"""LLM utilities for document processing workflows.

This module provides Anthropic Claude model integration with:
- Tiered model selection (Haiku/Sonnet/Opus)
- Extended thinking support for complex reasoning tasks
- Both synchronous and batch processing modes
"""

import asyncio
import json
import os
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from core.config import configure_langsmith

configure_langsmith()

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage


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
