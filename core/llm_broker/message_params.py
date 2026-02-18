"""Message parameter building and response parsing for the LLM Broker.

Provides pure functions for constructing Anthropic API parameters
and parsing response content. Used by both sync and batch execution paths.
"""

import json
import re
from typing import Any

from .schemas import LLMRequest

# Beta header for 1M context window
CONTEXT_1M_BETA = "context-1m-2025-08-07"


def sanitize_custom_id(identifier: str) -> str:
    """Convert identifier to valid Anthropic batch custom_id.

    The API requires custom_id to match pattern ^[a-zA-Z0-9_-]{1,64}$.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", identifier)
    return sanitized[:64]


def build_message_params(request: LLMRequest) -> dict[str, Any]:
    """Build Anthropic API message parameters from request.

    Args:
        request: The LLM request to build parameters for

    Returns:
        Dictionary of parameters for Anthropic messages API
    """
    params: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": request.messages if request.messages is not None else [{"role": "user", "content": request.prompt}],
    }

    if request.system:
        params["system"] = request.system

    if request.effort:
        params["thinking"] = {"type": "adaptive"}
        params["output_config"] = {"effort": request.effort}

    if request.tools:
        params["tools"] = request.tools
    if request.tool_choice:
        params["tool_choice"] = request.tool_choice

    return params


def parse_response_content(response: Any) -> tuple[Any, str | None]:
    """Parse content and thinking from Anthropic message response.

    Args:
        response: Anthropic message response

    Returns:
        Tuple of (content, thinking) where content is text or tool input
    """
    content = None
    thinking = None
    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            content = block.text
        elif block.type == "tool_use":
            tool_input = block.input
            # Unwrap $output wrapper if present
            if isinstance(tool_input, dict) and "$output" in tool_input and len(tool_input) == 1:
                tool_input = tool_input["$output"]
            content = json.dumps(tool_input)
    return content, thinking


def parse_response_content_with_blocks(
    response: Any,
) -> tuple[Any, str | None, list[dict[str, Any]]]:
    """Parse content, thinking, and raw content blocks from response.

    Extended version of parse_response_content() that also returns
    raw content blocks for tool-use scenarios where tool_use IDs
    are needed for multi-turn conversations.

    Returns:
        Tuple of (content, thinking, content_blocks)
    """
    content = None
    thinking = None
    blocks: list[dict[str, Any]] = []

    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
            blocks.append({"type": "thinking", "thinking": block.thinking})
        elif block.type == "text":
            content = block.text
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            tool_input = block.input
            if isinstance(tool_input, dict) and "$output" in tool_input and len(tool_input) == 1:
                tool_input = tool_input["$output"]
            content = json.dumps(tool_input)
            blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })

    return content, thinking, blocks
