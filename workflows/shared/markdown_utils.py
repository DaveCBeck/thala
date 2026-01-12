"""Markdown parsing utilities."""

import re


def extract_headings(markdown: str) -> list[dict]:
    """
    Extract all headings from markdown with their positions.

    Returns list of {level, text, position} dicts.
    """
    headings = []
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    for match in heading_pattern.finditer(markdown):
        level = len(match.group(1))
        text = match.group(2).strip()
        position = match.start()

        headings.append({
            "level": level,
            "text": text,
            "position": position,
        })

    return headings


def find_word_boundary(text: str, target_pos: int, direction: int = -1) -> int:
    """
    Find nearest word boundary near target position.

    Args:
        text: Full text
        target_pos: Target position
        direction: -1 for backward search, 1 for forward

    Returns:
        Position of word boundary
    """
    if direction == -1:
        # Search backward for whitespace
        pos = target_pos
        while pos > 0 and not text[pos].isspace():
            pos -= 1
        return pos
    else:
        # Search forward for whitespace
        pos = target_pos
        while pos < len(text) and not text[pos].isspace():
            pos += 1
        return pos
