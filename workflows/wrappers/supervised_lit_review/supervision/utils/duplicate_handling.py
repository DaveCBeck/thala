"""Duplicate section and header detection and handling utilities."""

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from workflows.wrappers.supervised_lit_review.supervision.types import SectionEditResult
from workflows.wrappers.supervised_lit_review.supervision.utils import SectionInfo

logger = logging.getLogger(__name__)


def detect_duplicate_sections(sections: list[SectionInfo]) -> list[tuple[str, str]]:
    """Detect sections with overlapping content that may cause duplicates."""
    duplicates = []
    for i, s1 in enumerate(sections):
        for s2 in sections[i + 1:]:
            ratio = SequenceMatcher(
                None,
                s1["section_content"][:500],
                s2["section_content"][:500]
            ).ratio()
            if ratio > 0.7:
                duplicates.append((s1["section_id"], s2["section_id"]))
    return duplicates


def detect_duplicate_headers(document: str) -> list[tuple[int, int, str]]:
    """Detect duplicate section headers in document."""
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
                    f"Duplicate header found: '{header_text}' at lines {positions[i] + 1} and {positions[i + 1] + 1}"
                )

    return duplicates


def remove_duplicate_headers(document: str, duplicates: list[tuple[int, int, str]]) -> str:
    """Remove duplicate section headers from document."""
    if not duplicates:
        return document

    lines = document.split("\n")
    sorted_dups = sorted(duplicates, key=lambda x: x[1], reverse=True)

    for line1, line2, header_text in sorted_dups:
        end_line = line2

        for i in range(line2 + 1, len(lines)):
            header_match = re.match(r"^(#{1,6})\s+(.+)$", lines[i].strip())
            if header_match:
                break
            end_line = i

        content_start1 = line1 + 1
        content_end1 = line2 - 1
        content_start2 = line2 + 1
        content_end2 = end_line

        content1 = "\n".join(lines[content_start1:content_end1 + 1]).strip()[:500]
        content2 = "\n".join(lines[content_start2:content_end2 + 1]).strip()[:500]

        similarity = SequenceMatcher(None, content1, content2).ratio()

        if similarity > 0.5:
            logger.info(
                f"Removing duplicate section at lines {line2 + 1}-{end_line + 1} "
                f"(similarity: {similarity:.2f})"
            )
            lines = lines[:line2] + lines[end_line + 1:]
        else:
            logger.info(
                f"Removing only duplicate header at line {line2 + 1} "
                f"(content similarity too low: {similarity:.2f})"
            )
            lines = lines[:line2] + lines[line2 + 1:]

    return "\n".join(lines)


def merge_duplicate_edits(
    section_results: dict[str, SectionEditResult],
    duplicates: list[tuple[str, str]],
) -> dict[str, SectionEditResult]:
    """Merge edits from duplicate sections, keeping the higher-confidence edit."""
    merged = dict(section_results)

    for id1, id2 in duplicates:
        if id1 in merged and id2 in merged:
            r1, r2 = merged[id1], merged[id2]
            if r2.confidence > r1.confidence:
                merged[id1] = SectionEditResult(
                    section_id=id1,
                    edited_content=r2.edited_content,
                    notes=f"{r1.notes}\n{r2.notes}".strip(),
                    new_paper_todos=r1.new_paper_todos + r2.new_paper_todos,
                    confidence=r2.confidence,
                )
            else:
                merged[id1] = SectionEditResult(
                    section_id=id1,
                    edited_content=r1.edited_content,
                    notes=f"{r1.notes}\n{r2.notes}".strip(),
                    new_paper_todos=r1.new_paper_todos + r2.new_paper_todos,
                    confidence=r1.confidence,
                )
            del merged[id2]

    return merged
