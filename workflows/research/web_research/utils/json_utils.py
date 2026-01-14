"""JSON parsing utilities."""

import json


def extract_json_from_llm_response(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown blocks and extra text.

    Handles:
    - Plain JSON
    - JSON wrapped in markdown code blocks (```json or ```)
    - JSON followed by extra explanatory text
    - JSON preceded by explanatory text
    """
    content = content.strip()

    # Remove markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
        # Find the closing ``` and extract content between
        end_idx = -1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        if end_idx > 0:
            content = "\n".join(lines[1:end_idx])
        else:
            content = "\n".join(lines[1:-1])
        content = content.strip()

    # Try direct parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Find JSON object by matching braces
    start_idx = content.find("{")
    if start_idx == -1:
        raise json.JSONDecodeError("No JSON object found", content, 0)

    # Count braces to find the complete JSON object
    depth = 0
    in_string = False
    escape_next = False
    end_idx = start_idx

    for i, char in enumerate(content[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if depth != 0:
        raise json.JSONDecodeError("Unbalanced braces in JSON", content, len(content))

    json_str = content[start_idx : end_idx + 1]
    return json.loads(json_str)
