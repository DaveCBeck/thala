"""
Metadata validation and normalization utilities.

Provides:
- Year validation and extraction
- Author name parsing and normalization
- Metadata merging with baseline preference
"""

import re
from typing import Any, Optional

from pydantic import BaseModel


# --- Year Validation ---


def extract_year(date_str: Optional[str]) -> Optional[int]:
    """
    Extract 4-digit year from date string.

    Returns None if no valid year found.
    Valid years: 1500-2099 (supports historical content)
    """
    if not date_str:
        return None
    # Match years 1500-1999 or 2000-2099
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", str(date_str))
    return int(match.group()) if match else None


def validate_year(year: Optional[str]) -> Optional[str]:
    """
    Validate year string is a valid 4-digit year.

    Returns the year as string if valid, None otherwise.
    """
    extracted = extract_year(year)
    return str(extracted) if extracted else None


# --- Author Name Parsing ---


class ParsedAuthorName(BaseModel):
    """Structured author name."""

    firstName: Optional[str] = None
    lastName: Optional[str] = None
    name: Optional[str] = None  # Single-field fallback (organizations, single names)

    def to_zotero_creator(self, creator_type: str = "author") -> dict:
        """Convert to Zotero creator dict."""
        if self.firstName and self.lastName:
            return {
                "firstName": self.firstName,
                "lastName": self.lastName,
                "creatorType": creator_type,
            }
        return {"name": self.name or "", "creatorType": creator_type}


# Name particles that should stay with the last name
NAME_PARTICLES = {
    "von",
    "van",
    "de",
    "der",
    "den",
    "la",
    "le",
    "di",
    "da",
    "dos",
    "das",
    "del",
    "della",
    "lo",
    "el",
    "al",
    "bin",
    "ibn",
    "af",
    "zu",
    "ter",
    "ten",
}


def parse_author_name(name: str) -> ParsedAuthorName:
    """
    Parse author name string into firstName/lastName.

    Handles:
    - "First Last" -> firstName="First", lastName="Last"
    - "First Middle Last" -> firstName="First Middle", lastName="Last"
    - "First M. Last" -> firstName="First M.", lastName="Last"
    - "A. B. LastName" -> firstName="A. B.", lastName="LastName"
    - "LastName, First" -> firstName="First", lastName="LastName"
    - "van der Berg" particles -> preserved in lastName
    - Single names -> uses `name` field
    - "Benjamin A. Black" (misplaced initial) -> firstName="Benjamin A.", lastName="Black"
    """
    name = name.strip()
    if not name:
        return ParsedAuthorName(name="Unknown")

    # Handle "Last, First" format (common in bibliographic data)
    if "," in name:
        parts = name.split(",", 1)
        last_name = parts[0].strip()
        first_name = parts[1].strip() if len(parts) > 1 else None
        if first_name:
            return ParsedAuthorName(firstName=first_name, lastName=last_name)
        return ParsedAuthorName(name=last_name)

    # Split on spaces
    parts = name.split()

    if len(parts) == 1:
        return ParsedAuthorName(name=parts[0])

    # Find where last name starts (accounting for particles)
    last_name_start = len(parts) - 1
    while last_name_start > 0:
        if parts[last_name_start - 1].lower() in NAME_PARTICLES:
            last_name_start -= 1
        else:
            break

    # Ensure at least one part goes to first name
    if last_name_start == 0:
        last_name_start = len(parts) - 1

    first_name = " ".join(parts[:last_name_start])
    last_name = " ".join(parts[last_name_start:])

    # Handle misplaced initials in last name
    # e.g., if we parsed "Benjamin A. Black" as firstName="Benjamin", lastName="A. Black"
    # The "A." should move to firstName
    last_parts = last_name.split()
    leading_initials = []
    actual_last_parts = []

    for part in last_parts:
        stripped = part.rstrip(".")
        # An initial is 1-2 uppercase chars, possibly with period
        if len(stripped) <= 2 and stripped.isupper() and not actual_last_parts:
            leading_initials.append(part)
        else:
            actual_last_parts.append(part)

    if leading_initials and actual_last_parts:
        first_name = (first_name + " " + " ".join(leading_initials)).strip()
        last_name = " ".join(actual_last_parts)

    return ParsedAuthorName(firstName=first_name, lastName=last_name)


def normalize_author_list(authors: list[str]) -> list[ParsedAuthorName]:
    """Parse and normalize a list of author name strings."""
    return [parse_author_name(name) for name in authors if name and name.strip()]


# --- Metadata Merging ---


def merge_metadata_with_baseline(
    baseline: dict[str, Any],
    extracted: dict[str, Any],
    baseline_priority_fields: set[str] | None = None,
) -> dict[str, Any]:
    """
    Merge extracted metadata with baseline (e.g., OpenAlex).

    Strategy: "Fill gaps only"
    - Baseline data (OpenAlex) is trusted for dates and authors
    - Extracted data fills in missing fields
    - Extraction can add data that baseline doesn't have

    Args:
        baseline: Trusted source data (e.g., from OpenAlex)
        extracted: LLM-extracted data from document
        baseline_priority_fields: Fields where baseline takes precedence if present
                                  Default: {"date", "authors", "publication_date", "year"}

    Returns:
        Merged metadata dict
    """
    if baseline_priority_fields is None:
        baseline_priority_fields = {"date", "authors", "publication_date", "year"}

    result = {}

    # Start with extracted values (non-empty only)
    for key, value in extracted.items():
        if value is not None and value != [] and value != {} and value != "":
            result[key] = value

    # Overlay baseline for priority fields (if baseline has non-empty value)
    for key in baseline_priority_fields:
        baseline_value = baseline.get(key)
        if baseline_value is not None and baseline_value != [] and baseline_value != "":
            result[key] = baseline_value

    # Fill gaps: if result doesn't have a field but baseline does, use baseline
    for key, value in baseline.items():
        if key not in result and value is not None and value != [] and value != "":
            result[key] = value

    return result
