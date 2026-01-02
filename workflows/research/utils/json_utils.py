"""JSON parsing utilities."""

import json


def extract_json_from_llm_response(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown blocks."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])
    return json.loads(content)
