"""LLM response extraction utilities."""


def extract_text_from_response(response) -> str:
    """Extract text from LLM response regardless of format."""
    if isinstance(response.content, str):
        return response.content.strip()
    if isinstance(response.content, list) and len(response.content) > 0:
        first_block = response.content[0]
        if isinstance(first_block, dict):
            return first_block.get("text", "").strip()
        if hasattr(first_block, "text"):
            return first_block.text.strip()
        return str(first_block).strip()
    return str(response.content).strip()
