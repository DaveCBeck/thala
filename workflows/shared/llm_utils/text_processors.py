"""Text processing functions using LLMs."""

import asyncio
import json
import logging
import os
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from .models import ModelTier, get_llm
from ..retry_utils import with_retry
from .response_parsing import extract_json_from_response

logger = logging.getLogger(__name__)


async def summarize_text(
    text: str,
    target_words: int = 100,
    context: Optional[str] = None,
    tier: ModelTier = ModelTier.SONNET,
) -> str:
    """Summarize text using Claude with retry logic."""
    llm = get_llm(tier=tier)

    prompt = f"Summarize the following text in approximately {target_words} words."
    if context:
        prompt += f"\n\nContext: {context}"
    prompt += f"\n\nText:\n{text}"

    async def _invoke():
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

    return await with_retry(_invoke)


async def extract_json(
    text: str,
    prompt: str,
    schema_hint: Optional[str] = None,
    tier: ModelTier = ModelTier.SONNET,
) -> dict[str, Any]:
    """Extract structured JSON from text using Claude.

    DEPRECATED: Consider using get_structured_output() instead for type-safe
    extraction with automatic strategy selection.
    """
    llm = get_llm(tier=tier)

    full_prompt = prompt
    if schema_hint:
        full_prompt += f"\n\nExpected schema:\n{schema_hint}"
    full_prompt += "\n\nRespond with ONLY valid JSON, no other text."
    full_prompt += f"\n\nText:\n{text}"

    async def _invoke():
        response = await llm.ainvoke([HumanMessage(content=full_prompt)])
        content = response.content.strip()
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


async def analyze_with_thinking(
    text: str,
    prompt: str,
    thinking_budget: int = 8000,
    tier: ModelTier = ModelTier.OPUS,
) -> tuple[str, Optional[str]]:
    """Perform complex analysis using extended thinking."""
    llm = get_llm(tier=tier, thinking_budget=thinking_budget, max_tokens=thinking_budget + 4096)

    full_prompt = f"{prompt}\n\nText:\n{text}"

    async def _invoke():
        response = await llm.ainvoke([HumanMessage(content=full_prompt)])

        thinking_content = None
        text_content = ""

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

    return await with_retry(_invoke)


async def extract_structured(
    text: str,
    prompt: str,
    schema: dict[str, Any],
    tier: ModelTier = ModelTier.SONNET,
) -> dict[str, Any]:
    """Extract structured data using Anthropic tool use (guaranteed valid JSON).

    DEPRECATED: Consider using get_structured_output() instead, which provides
    the same functionality with automatic strategy selection and retry logic.
    """
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    tool = {
        "name": "extract_data",
        "description": prompt,
        "input_schema": schema,
    }

    async def _invoke():
        response = await client.messages.create(
            model=tier.value,
            max_tokens=4096,
            tools=[tool],
            tool_choice={"type": "tool", "name": "extract_data"},
            messages=[
                {"role": "user", "content": f"{prompt}\n\nText:\n{text}"}
            ],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_data":
                return block.input

        raise RuntimeError("No tool use in response")

    return await with_retry(
        _invoke,
        error_message="Failed to extract structured data after {attempts} attempts"
    )
