"""Pre-flight citation validation for Loop4 edits.

Validates that section edits only reference papers already cited in the original section,
preventing citation drift where the LLM adds references to papers not in the corpus.
"""

import re
from typing import Set


def extract_citation_keys_from_text(text: str) -> Set[str]:
    """Extract all [@KEY] citation keys from text.

    Args:
        text: The text to search for citation keys

    Returns:
        Set of citation keys (without the @[] wrapper)

    Example:
        >>> extract_citation_keys_from_text("See [@smith2020] and [@jones2021]")
        {'smith2020', 'jones2021'}
    """
    pattern = r"\[@([^\]]+)\]"
    return set(re.findall(pattern, text))


def validate_edit_citations(
    original_section: str,
    edited_section: str,
    corpus_keys: Set[str],
) -> tuple[bool, list[str]]:
    """Validate that edited section only cites papers from corpus that were in original.

    This enforces the corpus-only citation policy for Loop4 edits:
    - New citations are only allowed if they reference papers in the corpus
    - Even then, we reject them since Loop4 should not add new citations

    Args:
        original_section: The original section content before editing
        edited_section: The section content after LLM editing
        corpus_keys: Set of valid zotero citation keys in the corpus

    Returns:
        Tuple of (is_valid, list of invalid citations with reasons)
        is_valid is True only if no new citations were added
    """
    original_citations = extract_citation_keys_from_text(original_section)
    edited_citations = extract_citation_keys_from_text(edited_section)

    # New citations added by the edit
    new_citations = edited_citations - original_citations

    # Check if new citations are in corpus (shouldn't be adding ANY new citations)
    invalid_citations = []
    for key in new_citations:
        if key not in corpus_keys:
            invalid_citations.append(f"{key} (not in corpus)")
        else:
            invalid_citations.append(f"{key} (new citation - not allowed in Loop4)")

    return len(invalid_citations) == 0, invalid_citations


def check_section_growth(
    original: str,
    edited: str,
    tolerance: float = 0.20,
) -> tuple[bool, float]:
    """Check if edited section is within tolerance of original length.

    Loop4 enforces a ±20% word count limit to prevent excessive expansion
    or reduction of section content.

    Args:
        original: Original section content
        edited: Edited section content
        tolerance: Maximum allowed growth/shrinkage ratio (default 0.20 = ±20%)

    Returns:
        Tuple of (is_within_tolerance, growth_ratio)
        growth_ratio is positive for expansion, negative for reduction
    """
    original_words = len(original.split())
    edited_words = len(edited.split())

    if original_words == 0:
        return True, 0.0

    growth_ratio = (edited_words - original_words) / original_words
    is_within_tolerance = abs(growth_ratio) <= tolerance
    return is_within_tolerance, growth_ratio
