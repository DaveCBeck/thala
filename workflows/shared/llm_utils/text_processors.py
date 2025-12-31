"""Text processing functions using LLMs."""

import asyncio
import json
import logging
import os
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from .models import ModelTier, get_llm

logger = logging.getLogger(__name__)


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


async def extract_structured(
    text: str,
    prompt: str,
    schema: dict[str, Any],
    tier: ModelTier = ModelTier.SONNET,
) -> dict[str, Any]:
    """
    Extract structured data using Anthropic tool use (guaranteed valid JSON).

    Uses tool use to force the model to return data matching the schema.
    This eliminates JSON parsing errors that can occur with prompt-based extraction.

    Args:
        text: Text to extract from
        prompt: Instructions for extraction
        schema: JSON schema dict defining the expected output structure.
                Must include "type", "properties", and optionally "required".
        tier: Model tier to use (default: SONNET)

    Returns:
        Extracted data as dict matching the schema

    Example:
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title"},
                "authors": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title"]
        }
        result = await extract_structured(text, "Extract metadata", schema)
    """
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    tool = {
        "name": "extract_data",
        "description": prompt,
        "input_schema": schema,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=tier.value,
                max_tokens=4096,
                tools=[tool],
                tool_choice={"type": "tool", "name": "extract_data"},
                messages=[
                    {"role": "user", "content": f"{prompt}\n\nText:\n{text}"}
                ],
            )

            # Extract tool use result
            for block in response.content:
                if block.type == "tool_use" and block.name == "extract_data":
                    return block.input

            raise RuntimeError("No tool use in response")

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                logger.warning(f"Structured extraction attempt {attempt + 1} failed: {e}")
            else:
                raise RuntimeError(f"Failed to extract structured data after {max_retries} attempts: {e}") from e
