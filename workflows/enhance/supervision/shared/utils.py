"""Utility functions for supervision loops."""

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def detect_duplicate_headers(document: str) -> list[tuple[int, int, str]]:
    """Detect duplicate section headers in document.

    Args:
        document: Markdown document text

    Returns:
        List of (first_line, second_line, header_text) tuples
    """
    lines = document.split("\n")
    header_positions: dict[str, list[int]] = {}

    for i, line in enumerate(lines):
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if header_match:
            header_text = header_match.group(0).strip().lower()
            if header_text not in header_positions:
                header_positions[header_text] = []
            header_positions[header_text].append(i)

    duplicates = []
    for header_text, positions in header_positions.items():
        if len(positions) > 1:
            for i in range(len(positions) - 1):
                duplicates.append((positions[i], positions[i + 1], header_text))
                logger.warning(
                    f"Duplicate header found: '{header_text}' at lines "
                    f"{positions[i] + 1} and {positions[i + 1] + 1}"
                )

    return duplicates


def remove_duplicate_headers(
    document: str, duplicates: list[tuple[int, int, str]]
) -> str:
    """Remove duplicate section headers from document.

    For each duplicate, removes the second occurrence and merges content
    if the sections have similar content.

    Args:
        document: Markdown document text
        duplicates: List of (first_line, second_line, header_text) tuples

    Returns:
        Document with duplicates removed
    """
    if not duplicates:
        return document

    lines = document.split("\n")
    sorted_dups = sorted(duplicates, key=lambda x: x[1], reverse=True)

    for line1, line2, header_text in sorted_dups:
        end_line = line2

        # Find where section 2 ends
        for i in range(line2 + 1, len(lines)):
            header_match = re.match(r"^(#{1,6})\s+(.+)$", lines[i].strip())
            if header_match:
                break
            end_line = i

        # Get content of both sections
        content_start1 = line1 + 1
        content_end1 = line2 - 1
        content_start2 = line2 + 1
        content_end2 = end_line

        content1 = "\n".join(lines[content_start1 : content_end1 + 1]).strip()[:500]
        content2 = "\n".join(lines[content_start2 : content_end2 + 1]).strip()[:500]

        similarity = SequenceMatcher(None, content1, content2).ratio()

        if similarity > 0.5:
            # High similarity - remove duplicate section entirely
            logger.info(
                f"Removing duplicate section '{header_text}' "
                f"(similarity: {similarity:.2f})"
            )
            del lines[line2 : end_line + 1]
        else:
            # Low similarity - just remove the header, keep content
            logger.info(
                f"Removing duplicate header '{header_text}' but keeping content "
                f"(similarity: {similarity:.2f})"
            )
            del lines[line2]

    return "\n".join(lines)
