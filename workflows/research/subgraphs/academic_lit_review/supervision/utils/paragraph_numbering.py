"""Paragraph numbering utilities for Loop 3 structural editing."""

import re


def number_paragraphs(doc: str) -> tuple[str, dict[int, str]]:
    """Add [P1], [P2] markers to each paragraph for structural editing.

    Paragraphs are separated by double newlines or markdown headers.
    Preserves markdown structure (headers, code blocks, lists).

    Returns:
        Tuple of (numbered_doc, paragraph_mapping) where mapping is {num: original_text}
    """
    lines = doc.split("\n")
    numbered_lines = []
    paragraph_mapping = {}
    paragraph_num = 1
    in_code_block = False
    current_paragraph_lines = []

    for line in lines:
        # Track code block boundaries
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        # Headers always start a new paragraph
        is_header = line.strip().startswith("#")

        # Empty line indicates paragraph boundary
        is_empty = line.strip() == ""

        if is_empty and not in_code_block and current_paragraph_lines:
            # End current paragraph
            paragraph_text = "\n".join(current_paragraph_lines)
            paragraph_mapping[paragraph_num] = paragraph_text
            numbered_lines.append(f"[P{paragraph_num}] {paragraph_text}")
            numbered_lines.append("")  # Preserve empty line
            paragraph_num += 1
            current_paragraph_lines = []
        elif is_header and not in_code_block:
            # Headers are standalone paragraphs
            if current_paragraph_lines:
                paragraph_text = "\n".join(current_paragraph_lines)
                paragraph_mapping[paragraph_num] = paragraph_text
                numbered_lines.append(f"[P{paragraph_num}] {paragraph_text}")
                paragraph_num += 1
                current_paragraph_lines = []

            paragraph_mapping[paragraph_num] = line
            numbered_lines.append(f"[P{paragraph_num}] {line}")
            paragraph_num += 1
        elif is_empty:
            numbered_lines.append(line)
        else:
            current_paragraph_lines.append(line)

    # Handle final paragraph
    if current_paragraph_lines:
        paragraph_text = "\n".join(current_paragraph_lines)
        paragraph_mapping[paragraph_num] = paragraph_text
        numbered_lines.append(f"[P{paragraph_num}] {paragraph_text}")

    return "\n".join(numbered_lines), paragraph_mapping


def strip_paragraph_numbers(doc: str) -> str:
    """Remove [Pn] markers from document."""
    # Match [P{number}] at start of lines, with optional whitespace
    return re.sub(r"^\[P\d+\]\s*", "", doc, flags=re.MULTILINE)
