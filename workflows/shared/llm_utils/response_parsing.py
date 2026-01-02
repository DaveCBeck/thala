"""LLM response parsing utilities."""

import json
from typing import Any, Optional


def extract_json_from_response(content: str, default: Optional[dict] = None) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    content = content.strip()

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if default is not None:
            return default
        raise


def extract_response_content(response: Any) -> str:
    """Extract text content from various LLM response formats."""
    if isinstance(response.content, str):
        return response.content.strip()
    if isinstance(response.content, list) and response.content:
        first_block = response.content[0]
        if isinstance(first_block, dict):
            return first_block.get("text", "").strip()
        if hasattr(first_block, "text"):
            return first_block.text.strip()
        return str(first_block).strip()
    return str(response.content).strip()
