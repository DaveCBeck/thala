"""Markdown parser for converting text to DocumentModel.

Parses markdown documents into structured Section/ContentBlock models
with stable IDs for reference during editing.
"""

import re
import logging
from typing import Literal

from .document_model import DocumentModel, Section, ContentBlock

logger = logging.getLogger(__name__)


def detect_block_type(content: str) -> Literal["paragraph", "list", "code", "quote", "table", "metadata"]:
    """Detect the type of content block."""
    stripped = content.strip()

    # Code block
    if stripped.startswith("```") or stripped.startswith("    "):
        return "code"

    # Quote block
    if stripped.startswith(">"):
        return "quote"

    # Table (has | characters in multiple lines)
    if "|" in stripped and stripped.count("\n") > 0:
        lines = stripped.split("\n")
        if sum(1 for line in lines if "|" in line) >= 2:
            return "table"

    # List (starts with -, *, or number.)
    if re.match(r"^[\-\*\d]+[\.\)]\s", stripped):
        return "list"

    # Metadata (italic lines with specific patterns like *Generated: ...*)
    if stripped.startswith("*") and stripped.endswith("*") and len(stripped) < 200:
        return "metadata"

    return "paragraph"


def parse_markdown_to_model(text: str) -> DocumentModel:
    """Parse markdown text into structured DocumentModel.

    Args:
        text: Raw markdown text

    Returns:
        DocumentModel with sections, blocks, and stable IDs
    """
    lines = text.split("\n")

    title = ""
    preamble_blocks: list[ContentBlock] = []
    sections: list[Section] = []
    section_stack: list[Section] = []  # Stack for tracking hierarchy
    current_paragraph_lines: list[str] = []
    in_code_block = False
    code_block_lines: list[str] = []

    def flush_paragraph():
        nonlocal current_paragraph_lines
        if current_paragraph_lines:
            content = "\n".join(current_paragraph_lines).strip()
            if content:
                block_type = detect_block_type(content)
                block = ContentBlock.from_content(content, block_type)
                if section_stack:
                    section_stack[-1].blocks.append(block)
                else:
                    preamble_blocks.append(block)
            current_paragraph_lines = []

    def flush_code_block():
        nonlocal code_block_lines, in_code_block
        if code_block_lines:
            content = "\n".join(code_block_lines)
            block = ContentBlock.from_content(content, "code")
            if section_stack:
                section_stack[-1].blocks.append(block)
            else:
                preamble_blocks.append(block)
            code_block_lines = []
        in_code_block = False

    for line in lines:
        # Handle code blocks
        if line.strip().startswith("```"):
            if in_code_block:
                code_block_lines.append(line)
                flush_code_block()
            else:
                flush_paragraph()
                in_code_block = True
                code_block_lines = [line]
            continue

        if in_code_block:
            code_block_lines.append(line)
            continue

        # Check for heading
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            # Handle title (first H1 before any sections)
            if level == 1 and not title and not sections:
                title = heading_text
                continue

            # Determine parent
            parent_id = None
            if section_stack:
                # Pop sections until we find one with lower level
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()
                if section_stack:
                    parent_id = section_stack[-1].section_id

            # Create new section
            new_section = Section.from_heading(heading_text, level, parent_id)

            # Add to structure
            if parent_id and section_stack:
                section_stack[-1].subsections.append(new_section)
            else:
                sections.append(new_section)

            section_stack.append(new_section)
            continue

        # Handle horizontal rule
        if re.match(r"^[\-\*\_]{3,}\s*$", line):
            flush_paragraph()
            continue

        # Empty line - potential paragraph break
        if not line.strip():
            # Only flush if we have content (avoid multiple empty lines)
            if current_paragraph_lines:
                flush_paragraph()
            continue

        # Regular content line
        current_paragraph_lines.append(line)

    # Flush any remaining content
    if in_code_block:
        flush_code_block()
    flush_paragraph()

    model = DocumentModel(
        title=title,
        sections=sections,
        preamble_blocks=preamble_blocks,
    )

    logger.info(
        f"Parsed document: {model.total_words} words, "
        f"{model.section_count} sections, {model.block_count} blocks"
    )

    return model


def validate_document_model(model: DocumentModel) -> list[str]:
    """Validate document model integrity.

    Returns:
        List of validation warnings (empty if valid)
    """
    warnings = []

    # Check for empty sections
    for section in model.get_all_sections():
        if not section.blocks and not section.subsections:
            warnings.append(f"Empty section: {section.heading} ({section.section_id})")

    # Check for very short blocks (might be parsing errors)
    for block_id, (block, _) in model._block_index.items():
        if block.word_count < 3 and block.block_type == "paragraph":
            warnings.append(f"Very short paragraph block: {block_id}")

    # Check for duplicate IDs (shouldn't happen with content hashing)
    seen_section_ids = set()
    for section in model.get_all_sections():
        if section.section_id in seen_section_ids:
            warnings.append(f"Duplicate section ID: {section.section_id}")
        seen_section_ids.add(section.section_id)

    return warnings
