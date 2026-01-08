"""Section splitting utilities for Loop 4 parallel editing."""

import re
from typing_extensions import TypedDict
import tiktoken


class SectionInfo(TypedDict):
    """Information about a document section."""

    section_id: str
    section_content: str
    heading_level: int
    start_line: int
    end_line: int


def split_into_sections(doc: str, max_tokens: int = 5000) -> list[SectionInfo]:
    """Split document into sections for parallel editing.

    Strategy:
    - Split by major headings (## or #) if sections are < max_tokens
    - If a section exceeds max_tokens, split by subheadings (###)
    - Each section gets a unique section_id based on heading

    Args:
        doc: Full document text
        max_tokens: Maximum tokens per section (uses cl100k_base encoding)

    Returns:
        List of SectionInfo dicts with section metadata
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    lines = doc.split("\n")
    sections = []

    # First pass: identify all heading boundaries
    heading_positions = []
    for i, line in enumerate(lines):
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            heading_positions.append((i, level, title))

    if not heading_positions:
        # No headings, treat whole document as one section
        return [
            SectionInfo(
                section_id="full_document",
                section_content=doc,
                heading_level=0,
                start_line=0,
                end_line=len(lines) - 1,
            )
        ]

    # Second pass: create sections
    for idx, (line_num, level, title) in enumerate(heading_positions):
        # Determine section end
        if idx + 1 < len(heading_positions):
            end_line = heading_positions[idx + 1][0] - 1
        else:
            end_line = len(lines) - 1

        section_content = "\n".join(lines[line_num : end_line + 1])
        token_count = len(encoding.encode(section_content))

        # Generate section_id from title
        section_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")

        # If section is too large and not a level 3 heading, try to split further
        if token_count > max_tokens and level < 3:
            # Find subsections within this section
            subsections = _split_large_section(
                lines[line_num : end_line + 1],
                line_num,
                section_id,
                level,
                max_tokens,
                encoding,
            )
            sections.extend(subsections)
        else:
            sections.append(
                SectionInfo(
                    section_id=section_id,
                    section_content=section_content,
                    heading_level=level,
                    start_line=line_num,
                    end_line=end_line,
                )
            )

    return sections


def _split_large_section(
    section_lines: list[str],
    base_line_num: int,
    parent_id: str,
    parent_level: int,
    max_tokens: int,
    encoding,
) -> list[SectionInfo]:
    """Split a large section by its subheadings."""
    subsections = []
    subheading_positions = []

    for i, line in enumerate(section_lines):
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if header_match:
            level = len(header_match.group(1))
            if level > parent_level:
                title = header_match.group(2).strip()
                subheading_positions.append((i, level, title))

    if not subheading_positions:
        # No subheadings, return entire section as-is
        return [
            SectionInfo(
                section_id=parent_id,
                section_content="\n".join(section_lines),
                heading_level=parent_level,
                start_line=base_line_num,
                end_line=base_line_num + len(section_lines) - 1,
            )
        ]

    for idx, (line_offset, level, title) in enumerate(subheading_positions):
        if idx + 1 < len(subheading_positions):
            end_offset = subheading_positions[idx + 1][0] - 1
        else:
            end_offset = len(section_lines) - 1

        content = "\n".join(section_lines[line_offset : end_offset + 1])
        section_id = f"{parent_id}_{re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')}"

        subsections.append(
            SectionInfo(
                section_id=section_id,
                section_content=content,
                heading_level=level,
                start_line=base_line_num + line_offset,
                end_line=base_line_num + end_offset,
            )
        )

    return subsections
