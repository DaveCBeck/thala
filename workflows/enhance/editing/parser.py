"""Markdown parser for converting text to DocumentModel.

Parses markdown documents into structured Section/ContentBlock models
with stable IDs for reference during editing.
"""

import re
import logging
from typing import Literal

from .document_model import DocumentModel, Section, ContentBlock

logger = logging.getLogger(__name__)


def _normalize_heading(text: str) -> str:
    """Normalize heading for comparison (matches document_model version)."""
    text = text.lower()
    # Strip leading section/chapter numbers: "1.", "1.2.", "Chapter 1:", "Section 2.3"
    text = re.sub(r'^(?:chapter|section)?\s*[\d.]+[.:)]*\s*', '', text)
    # Remove remaining non-alphabetic characters
    return re.sub(r'[^a-z]', '', text)


def _merge_duplicate_sections(model: DocumentModel) -> int:
    """Merge sections with matching normalized headings.

    Handles malformed documents where the same section appears twice, e.g.:
        ## 1. Introduction   (empty, level 2)
        # Introduction       (with content, level 1)

    The content is consolidated into one section and duplicates are removed.

    Returns:
        Number of sections merged/removed
    """
    merge_count = 0

    def flatten_all_sections(sections: list[Section]) -> list[Section]:
        """Get all sections in document order, flattened."""
        result = []
        for section in sections:
            result.append(section)
            result.extend(flatten_all_sections(section.subsections))
        return result

    def merge_in_list(sections: list[Section]) -> list[Section]:
        """Process a list of sections, merging duplicates at same level."""
        nonlocal merge_count

        if not sections:
            return sections

        # First, recursively process subsections
        for section in sections:
            section.subsections = merge_in_list(section.subsections)

        # Now merge duplicates at this level
        heading_map: dict[str, int] = {}  # normalized_heading -> index in result
        result: list[Section] = []

        for section in sections:
            norm_heading = _normalize_heading(section.heading)

            if norm_heading in heading_map:
                # Duplicate found - merge into the existing section
                existing_idx = heading_map[norm_heading]
                existing = result[existing_idx]

                # Merge blocks (append to existing)
                existing.blocks.extend(section.blocks)

                # Merge subsections (append, then recursively dedupe)
                existing.subsections.extend(section.subsections)
                existing.subsections = merge_in_list(existing.subsections)

                logger.debug(
                    f"Merged duplicate section '{section.heading}' into '{existing.heading}'"
                )
                merge_count += 1
            else:
                # New heading - add to result
                heading_map[norm_heading] = len(result)
                result.append(section)

        return result

    # First pass: merge at same level
    model.sections = merge_in_list(model.sections)

    # Second pass: remove empty numbered sections that have a matching content section
    # This handles the case where ## 1. Introduction (empty) is followed by # Introduction (with content)
    all_sections = flatten_all_sections(model.sections)
    headings_with_content = {
        _normalize_heading(s.heading)
        for s in all_sections
        if s.blocks or s.subsections
    }

    def remove_empty_duplicates(sections: list[Section]) -> list[Section]:
        """Remove empty sections whose heading matches a section with content."""
        nonlocal merge_count
        result = []

        for section in sections:
            # Recursively process subsections first
            section.subsections = remove_empty_duplicates(section.subsections)

            norm_heading = _normalize_heading(section.heading)
            is_empty = not section.blocks and not section.subsections
            has_content_elsewhere = norm_heading in headings_with_content

            if is_empty and has_content_elsewhere:
                # This empty section has a content-bearing duplicate elsewhere
                logger.debug(
                    f"Removing empty duplicate section '{section.heading}'"
                )
                merge_count += 1
            else:
                result.append(section)

        return result

    model.sections = remove_empty_duplicates(model.sections)

    # Third pass: remove sections that just duplicate the document title
    if model.title:
        title_norm = _normalize_heading(model.title)

        def remove_title_duplicates(sections: list[Section]) -> list[Section]:
            """Remove sections whose heading matches the document title."""
            nonlocal merge_count
            result = []
            for section in sections:
                section.subsections = remove_title_duplicates(section.subsections)
                if _normalize_heading(section.heading) == title_norm:
                    # This section duplicates the title - merge its content into preamble
                    model.preamble_blocks.extend(section.blocks)
                    # Add any subsections as top-level sections
                    for sub in section.subsections:
                        sub.parent_id = None
                    result.extend(section.subsections)
                    logger.debug(f"Removed title-duplicate section '{section.heading}'")
                    merge_count += 1
                else:
                    result.append(section)
            return result

        model.sections = remove_title_duplicates(model.sections)

    return merge_count


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

    # Post-process: merge duplicate sections with matching normalized headings
    # This handles malformed documents with both "## 1. Introduction" and "# Introduction"
    merge_count = _merge_duplicate_sections(model)
    if merge_count > 0:
        logger.info(f"Merged {merge_count} duplicate sections during parsing")
        model._build_indexes()  # Rebuild after modifications

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
