"""Citation parsing utilities."""

import re
from typing import Set


def extract_citation_keys_from_text(text: str) -> Set[str]:
    """Extract all [@KEY] citation keys from text."""
    pattern = r"\[@([^\]]+)\]"
    return set(re.findall(pattern, text))


def strip_invalid_citations(
    text: str,
    invalid_keys: Set[str],
    add_todo: bool = True,
) -> str:
    """Remove or mark invalid citations in text."""
    result = text
    for key in invalid_keys:
        pattern = rf"\[@{re.escape(key)}\]"
        if add_todo:
            replacement = f"<!-- TODO: Verify citation [@{key}] - not found in Zotero -->"
        else:
            replacement = ""
        result = re.sub(pattern, replacement, result)
    return result


def check_section_growth(
    original: str,
    edited: str,
    tolerance: float = 0.20,
) -> tuple[bool, float]:
    """Check if edited section is within tolerance of original length."""
    original_words = len(original.split())
    edited_words = len(edited.split())

    if original_words == 0:
        return True, 0.0

    growth_ratio = (edited_words - original_words) / original_words
    is_within_tolerance = abs(growth_ratio) <= tolerance
    return is_within_tolerance, growth_ratio


def is_plausible_citation_key(key: str, valid_keys: Set[str]) -> bool:
    """Check if a key could plausibly match a valid citation."""
    key_lower = key.lower()

    for valid_key in valid_keys:
        if valid_key.lower() == key_lower:
            return True

    key_normalized = re.sub(r"[_\-]", "", key_lower)
    for valid_key in valid_keys:
        valid_normalized = re.sub(r"[_\-]", "", valid_key.lower())
        if key_normalized == valid_normalized:
            return True

    return False


def generate_fallback_key(doi: str, summary: dict) -> str:
    """Generate a fallback Zotero citation key from DOI or metadata."""
    authors = summary.get("authors", [])
    year = summary.get("year")

    if authors and year:
        first_author = authors[0]
        if "," in first_author:
            surname = first_author.split(",")[0].strip()
        else:
            surname = first_author.split()[-1] if first_author.split() else ""

        if surname:
            surname_clean = re.sub(r"[^a-zA-Z]", "", surname).upper()[:8]
            return f"{surname_clean}{year}"

    doi_suffix = doi.split("/")[-1] if "/" in doi else doi
    clean_suffix = re.sub(r"[^a-zA-Z0-9]", "", doi_suffix).upper()[:12]
    return clean_suffix if clean_suffix else "UNKNOWN"
