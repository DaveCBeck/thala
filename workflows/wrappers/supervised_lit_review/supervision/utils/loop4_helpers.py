"""Helper utilities for Loop 4 section editing.

Contains:
- Metadata stripping for documents
- Placeholder detection for section edits
- Anchor-based section replacement
- Content-hash-based stable references (per best practices)

Best Practice Reference:
    .context/best-practice-llm-structural-document-editing-20260116.md
"""

import hashlib
import logging
import re
from difflib import SequenceMatcher
from typing import Optional

from .section_splitting import SectionInfo

logger = logging.getLogger(__name__)


# =============================================================================
# Content-Hash-Based Stable References (Best Practice Pattern)
# =============================================================================


def compute_content_hash(content: str, chars: int = 100) -> str:
    """Compute a short hash of content for stable identification.

    Per best practices: use first N chars to create stable ID that
    survives minor edits but detects major changes.

    Args:
        content: Content to hash
        chars: Number of leading characters to hash (default 100)

    Returns:
        8-character hex hash
    """
    # Normalize whitespace for more stable hashing
    normalized = " ".join(content[:chars].split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:8]


def validate_content_unchanged(
    document: str,
    section: SectionInfo,
    expected_hash: str,
) -> tuple[bool, str]:
    """Validate that section content hasn't changed since edit was planned.

    Per best practices: check content_hash before applying edits to ensure
    the target hasn't been modified by concurrent edits.

    Args:
        document: Current document text
        section: Section to validate
        expected_hash: Hash when edit was planned

    Returns:
        Tuple of (is_valid, current_content)
    """
    heading = get_section_heading(section)
    if not heading:
        return False, ""

    start, end = find_section_by_heading(document, heading, section["heading_level"])
    if start < 0:
        return False, ""

    lines = document.split("\n")
    current_content = "\n".join(lines[start:end + 1])
    current_hash = compute_content_hash(current_content)

    if current_hash != expected_hash:
        logger.warning(
            f"Content hash mismatch for section '{section['section_id']}': "
            f"expected {expected_hash}, got {current_hash}"
        )
        return False, current_content

    return True, current_content


def create_section_reference(section: SectionInfo) -> dict:
    """Create a stable reference for a section.

    Per best practices: content-based IDs are more stable than line numbers.

    Args:
        section: Section to reference

    Returns:
        Dict with stable reference fields
    """
    return {
        "section_id": section["section_id"],
        "content_hash": compute_content_hash(section["section_content"]),
        "heading": get_section_heading(section),
        "heading_level": section["heading_level"],
    }


# =============================================================================
# Metadata Handling
# =============================================================================


# Patterns that indicate placeholder/error content in section edits
PLACEHOLDER_PATTERNS = [
    r"(?i)section\s+(is\s+)?empty",
    r"(?i)content\s+status:\s*section\s+empty",
    r"(?i)no\s+substantive\s+content",
    r"(?i)please\s+provide",
    r"(?i)this\s+section\s+(is\s+)?(currently\s+)?empty",
    r"(?i)unable\s+to\s+edit",
    r"(?i)cannot\s+edit\s+(this\s+)?section",
    r"(?i)\[placeholder\]",
    r"(?i)\[content\s+needed\]",
    r"(?i)\[to\s+be\s+added\]",
]


def strip_document_metadata(document: str) -> tuple[str, str]:
    """Strip metadata header from document, returning (stripped_doc, metadata).

    Handles the standard lit review metadata format:
    ```
    # Literature Review (After Loop N: ...): Topic

    *Generated: YYYY-MM-DD HH:MM:SS*

    *quality: X | language: Y | ...*

    ---
    ```

    Args:
        document: Full document text

    Returns:
        Tuple of (stripped_document, metadata_block)
        If no metadata found, returns (document, "")
    """
    lines = document.split("\n")
    metadata_end_idx = 0
    found_title = False
    found_generated = False
    found_separator = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Look for title line with Loop info
        if stripped.startswith("# ") and ("Loop" in stripped or "Literature Review" in stripped):
            found_title = True
            continue

        # Look for generated timestamp
        if stripped.startswith("*Generated:") or stripped.startswith("*quality:"):
            found_generated = True
            continue

        # Look for horizontal rule separator
        if stripped == "---" and found_title:
            found_separator = True
            metadata_end_idx = i + 1
            # Skip any blank lines after separator
            while metadata_end_idx < len(lines) and not lines[metadata_end_idx].strip():
                metadata_end_idx += 1
            break

        # If we hit actual content without finding the pattern, stop
        if stripped and not stripped.startswith("*") and not stripped.startswith("#"):
            if found_title and found_generated:
                # Partial metadata - find the separator
                continue
            else:
                # No proper metadata header
                break

    if found_separator and metadata_end_idx > 0:
        metadata = "\n".join(lines[:metadata_end_idx])
        stripped_doc = "\n".join(lines[metadata_end_idx:])
        logger.debug(f"Stripped {metadata_end_idx} lines of metadata from document")
        return stripped_doc.strip(), metadata.strip()

    return document, ""


def restore_document_metadata(document: str, metadata: str) -> str:
    """Restore metadata header to document.

    Args:
        document: Document content (without metadata)
        metadata: Metadata block to prepend

    Returns:
        Document with metadata restored
    """
    if not metadata:
        return document
    return f"{metadata}\n\n{document}"


def detect_placeholder_content(content: str, threshold: float = 0.3) -> bool:
    """Detect if content appears to be placeholder/error text.

    Checks for:
    1. Known placeholder patterns
    2. Very short content with error-like language
    3. Content that's mostly instructions rather than actual review text

    Args:
        content: Section content to check
        threshold: Minimum ratio of content that must be non-placeholder (0-1)

    Returns:
        True if content appears to be a placeholder, False otherwise
    """
    content_lower = content.lower()

    # Check for explicit placeholder patterns
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, content):
            logger.debug(f"Placeholder pattern detected: {pattern}")
            return True

    # Check for very short content that looks like an error message
    if len(content.strip()) < 500:
        error_indicators = [
            "cannot", "unable", "empty", "missing", "provide", "needed",
            "placeholder", "to be added", "content required"
        ]
        indicator_count = sum(1 for ind in error_indicators if ind in content_lower)
        if indicator_count >= 2:
            logger.debug(f"Short content with {indicator_count} error indicators")
            return True

    # Check for content that's mostly markdown headers with no substance
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if lines:
        header_lines = sum(1 for l in lines if l.startswith("#"))
        # If more than 50% of non-empty lines are headers, suspicious
        if header_lines / len(lines) > 0.5 and len(lines) < 10:
            logger.debug(f"Content is mostly headers ({header_lines}/{len(lines)} lines)")
            return True

    return False


def validate_section_edit(
    original_content: str,
    edited_content: str,
    section_id: str,
) -> tuple[bool, str]:
    """Validate a section edit for quality issues.

    Checks:
    1. Not placeholder content
    2. Not drastically shorter than original (unless original was tiny)
    3. Maintains basic structure

    Args:
        original_content: Original section content
        edited_content: Edited section content
        section_id: Section identifier for logging

    Returns:
        Tuple of (is_valid, reason) where reason explains rejection
    """
    # Check for placeholder content
    if detect_placeholder_content(edited_content):
        return False, f"Section '{section_id}' edit contains placeholder/error content"

    # Check for drastic shrinkage (unless original was very short)
    original_words = len(original_content.split())
    edited_words = len(edited_content.split())

    if original_words > 50:  # Only check if original had substance
        shrinkage = 1 - (edited_words / original_words) if original_words > 0 else 0
        if shrinkage > 0.8:  # More than 80% reduction
            return False, f"Section '{section_id}' edit lost {shrinkage*100:.0f}% of content"

    return True, ""


def find_section_by_heading(
    document: str,
    heading_text: str,
    heading_level: int,
) -> tuple[int, int]:
    """Find a section in document by its heading text.

    Uses fuzzy matching to handle minor heading changes during editing.

    Args:
        document: Full document text
        heading_text: The heading text to find (without # prefix)
        heading_level: The heading level (1-6)

    Returns:
        Tuple of (start_line, end_line) or (-1, -1) if not found
    """
    lines = document.split("\n")
    heading_pattern = f"^{'#' * heading_level}\\s+"

    best_match_idx = -1
    best_match_ratio = 0.0

    for i, line in enumerate(lines):
        if re.match(heading_pattern, line.strip()):
            # Extract the heading text from the line
            line_heading = re.sub(r"^#+\s+", "", line.strip())

            # Fuzzy match
            ratio = SequenceMatcher(None, heading_text.lower(), line_heading.lower()).ratio()
            if ratio > best_match_ratio and ratio > 0.7:
                best_match_ratio = ratio
                best_match_idx = i

    if best_match_idx == -1:
        return -1, -1

    # Find end of section (next heading of same or higher level, or end of doc)
    end_idx = len(lines) - 1
    for i in range(best_match_idx + 1, len(lines)):
        header_match = re.match(r"^(#{1,6})\s+", lines[i].strip())
        if header_match:
            match_level = len(header_match.group(1))
            if match_level <= heading_level:
                end_idx = i - 1
                break

    return best_match_idx, end_idx


def replace_section_by_anchor(
    document: str,
    section: SectionInfo,
    new_content: str,
) -> tuple[str, bool]:
    """Replace a section in document using heading anchor matching.

    More robust than line-number based replacement when document
    structure has changed.

    Args:
        document: Full document text
        section: Section info with heading details
        new_content: New content to replace section with

    Returns:
        Tuple of (updated_document, success)
    """
    # Extract heading text from section content
    first_line = section["section_content"].split("\n")[0].strip()
    heading_match = re.match(r"^(#{1,6})\s+(.+)$", first_line)

    if not heading_match:
        logger.warning(f"Section '{section['section_id']}' has no valid heading")
        return document, False

    heading_level = len(heading_match.group(1))
    heading_text = heading_match.group(2)

    start_line, end_line = find_section_by_heading(document, heading_text, heading_level)

    if start_line == -1:
        logger.warning(
            f"Could not find section '{section['section_id']}' by heading: {heading_text}"
        )
        return document, False

    lines = document.split("\n")

    # Replace the section
    new_lines = new_content.split("\n")
    updated_lines = lines[:start_line] + new_lines + lines[end_line + 1:]

    logger.debug(
        f"Replaced section '{section['section_id']}' at lines {start_line}-{end_line} "
        f"with {len(new_lines)} lines"
    )

    return "\n".join(updated_lines), True


def build_stable_section_map(sections: list[SectionInfo]) -> dict[str, SectionInfo]:
    """Build a stable section map keyed by section_id.

    This map should be created once at the start of Loop 4 and used
    throughout all iterations to maintain stable section references.

    Args:
        sections: List of sections from initial split

    Returns:
        Dict mapping section_id to SectionInfo
    """
    return {s["section_id"]: s for s in sections}


def get_section_heading(section: SectionInfo) -> str:
    """Extract the heading text from a section.

    Args:
        section: Section info

    Returns:
        Heading text (without # prefix) or empty string
    """
    first_line = section["section_content"].split("\n")[0].strip()
    heading_match = re.match(r"^#{1,6}\s+(.+)$", first_line)
    return heading_match.group(1) if heading_match else ""
