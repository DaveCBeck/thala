---
module: document_processing
date: 2026-01-26
problem_type: llm_issue
component: structured_output
symptoms:
  - "Year 2030 is in the future validation errors"
  - "Author names stored as single string instead of first/last"
  - "Name particles like 'van der Berg' split incorrectly"
  - "LLM-extracted dates overwriting correct OpenAlex dates"
root_cause: prompt_failure
resolution_type: code_fix
severity: high
tags: [metadata, validation, author-parsing, year-extraction, llm-hallucination, openalex, merge-strategy]
---

# Metadata Parsing and Validation Utilities

## Problem

LLM-extracted metadata from documents is often unreliable:

1. **Year hallucination**: LLMs extract non-existent years (e.g., "2030", "1200") or malformed dates
2. **Author name inconsistency**: Names come in varied formats that break downstream systems expecting structured firstName/lastName
3. **Overwriting trusted data**: LLM-extracted metadata overwrites reliable baseline data from OpenAlex

### Symptoms

```python
# Hallucinated year
metadata = {"date": "Published sometime around 2030"}  # Future year

# Inconsistent author formats
authors = [
    "John Smith",           # First Last
    "Smith, John",          # Last, First
    "J. R. R. Tolkien",     # Multiple initials
    "Ludwig van Beethoven", # Name particle
    "Benjamin A. Black",    # Misplaced initial
]

# LLM overwrites OpenAlex
openalex_date = "2023-05-15"  # Reliable
llm_extracted = "sometime in 2023"  # Unreliable but used
```

### Impact

- Invalid data stored in Zotero (future years, malformed names)
- Name particles incorrectly split: "van Beethoven" becomes lastName="Beethoven"
- Trusted OpenAlex metadata replaced with unreliable LLM extractions
- Downstream citation formatting failures

## Root Cause

**LLM extraction limitations**: Language models extract metadata from document text but:
- Cannot reliably validate years against current date
- Parse names inconsistently based on context
- May hallucinate missing information
- Have no concept of "trusted" vs "unreliable" sources

**Missing validation layer**: No validation between LLM output and database storage.

**No merge strategy**: Extracted data blindly overwrites existing metadata without considering source reliability.

## Solution

Three complementary utilities in `workflows/shared/metadata_utils.py`:

### 1. Year Validation (`extract_year`, `validate_year`)

Regex-based extraction with range validation:

```python
import re
from typing import Optional


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
```

**Key design decisions:**

| Decision | Rationale |
|----------|-----------|
| Range 1500-2099 | Supports historical academic content; rejects distant past/future |
| Regex `\b(1[5-9]\d{2}\|20\d{2})\b` | Word boundaries prevent matching partial numbers |
| Return None for invalid | Fail safely rather than store bad data |

**Usage in Pydantic model:**

```python
from pydantic import BaseModel, model_validator

class DocumentMetadata(BaseModel):
    year: Optional[str] = Field(default=None, description="Publication year (4-digit)")
    date: Optional[str] = Field(default=None, description="Full publication date")

    @model_validator(mode="after")
    def ensure_valid_year(self) -> "DocumentMetadata":
        """Ensure year field has valid value; extract from date if needed."""
        if self.year:
            extracted = extract_year(self.year)
            self.year = str(extracted) if extracted else None

        # Fall back to extracting year from date
        if not self.year and self.date:
            extracted = extract_year(self.date)
            self.year = str(extracted) if extracted else None

        return self
```

### 2. Author Name Parsing (`parse_author_name`)

Handles multiple name formats and preserves particles:

```python
from pydantic import BaseModel

# Name particles that should stay with the last name
NAME_PARTICLES = {
    "von", "van", "de", "der", "den", "la", "le", "di", "da",
    "dos", "das", "del", "della", "lo", "el", "al", "bin", "ibn",
    "af", "zu", "ter", "ten",
}


class ParsedAuthorName(BaseModel):
    """Structured author name."""
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    name: Optional[str] = None  # Single-field fallback

    def to_zotero_creator(self, creator_type: str = "author") -> dict:
        """Convert to Zotero creator dict."""
        if self.firstName and self.lastName:
            return {
                "firstName": self.firstName,
                "lastName": self.lastName,
                "creatorType": creator_type,
            }
        return {"name": self.name or "", "creatorType": creator_type}


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
    # e.g., "Benjamin A. Black" initially parsed as firstName="Benjamin", lastName="A. Black"
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
```

**Parsing examples:**

| Input | firstName | lastName |
|-------|-----------|----------|
| `"John Smith"` | John | Smith |
| `"John Michael Smith"` | John Michael | Smith |
| `"J. R. R. Tolkien"` | J. R. R. | Tolkien |
| `"Smith, John"` | John | Smith |
| `"Ludwig van Beethoven"` | Ludwig | van Beethoven |
| `"Johannes van der Waals"` | Johannes | van der Waals |
| `"Benjamin A. Black"` | Benjamin A. | Black |
| `"Madonna"` | - | - (name="Madonna") |

### 3. Metadata Merging (`merge_metadata_with_baseline`)

Strategy: **"Fill gaps only"** - baseline (OpenAlex) takes priority for critical fields:

```python
from typing import Any


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
```

**Merge behavior:**

```python
baseline = {
    "date": "2023-05-15",       # Trusted (OpenAlex)
    "authors": ["John Smith"],   # Trusted
    "venue": "Nature",           # Trusted
}

extracted = {
    "date": "sometime in 2023",  # Unreliable (LLM)
    "authors": ["J. Smith"],     # Less reliable
    "publisher": "Academic Press",  # New data
    "isbn": "978-0-123456-78-9",    # New data
}

result = merge_metadata_with_baseline(baseline, extracted)
# Result:
# {
#     "date": "2023-05-15",      # Baseline wins (priority field)
#     "authors": ["John Smith"],  # Baseline wins (priority field)
#     "venue": "Nature",          # Baseline fills gap
#     "publisher": "Academic Press",  # Extracted adds new
#     "isbn": "978-0-123456-78-9",    # Extracted adds new
# }
```

## Integration Example

Full integration in `update_zotero.py`:

```python
from workflows.shared.metadata_utils import (
    merge_metadata_with_baseline,
    parse_author_name,
    validate_year,
)


def _get_baseline_metadata(state: DocumentProcessingState) -> dict[str, Any]:
    """Extract baseline metadata from OpenAlex/extra_metadata."""
    doc_input = state.get("input", {})
    extra_metadata = doc_input.get("extra_metadata", {})

    # Convert OpenAlex authors format if needed
    baseline_authors = extra_metadata.get("authors", [])
    if baseline_authors and isinstance(baseline_authors[0], dict):
        baseline_authors = [a.get("name", "") for a in baseline_authors if a.get("name")]

    return {
        "authors": baseline_authors,
        "date": extra_metadata.get("publication_date") or extra_metadata.get("date"),
        "year": extra_metadata.get("year"),
        "title": extra_metadata.get("title") or doc_input.get("title"),
        "publisher": extra_metadata.get("publisher"),
    }


async def update_zotero(state: DocumentProcessingState) -> dict[str, Any]:
    """Update Zotero with validated, merged metadata."""
    metadata_updates = state.get("metadata_updates", {})

    # Get baseline from OpenAlex
    baseline = _get_baseline_metadata(state)

    # Merge: OpenAlex preferred for dates/authors
    merged = merge_metadata_with_baseline(baseline, metadata_updates)

    # Validate year before storing
    date_to_validate = merged.get("year") or merged.get("date")
    if date_to_validate:
        validated_year = validate_year(date_to_validate)
        if validated_year:
            fields["date"] = validated_year
        else:
            logger.warning(f"Invalid year value, not updating: {date_to_validate}")

    # Parse author names properly
    if "authors" in merged and merged["authors"]:
        creators = []
        for author_name in merged["authors"]:
            if not author_name or not author_name.strip():
                continue
            parsed = parse_author_name(author_name)
            creators.append(
                ZoteroCreator(
                    firstName=parsed.firstName,
                    lastName=parsed.lastName,
                    name=parsed.name,
                    creatorType="author",
                )
            )
        if creators:
            update.creators = creators
```

## Files Modified

- `workflows/shared/metadata_utils.py` - New file with validation utilities
- `workflows/shared/__init__.py` - Export new utilities
- `workflows/document_processing/nodes/metadata_agent.py` - Added year field and Pydantic validator
- `workflows/document_processing/nodes/update_zotero.py` - Use proper name parsing and baseline merge
- `testing/test_metadata_utils.py` - Comprehensive unit tests (38 tests)

## Testing

The solution includes comprehensive tests covering edge cases:

```python
class TestYearExtraction:
    def test_year_from_full_date(self):
        assert extract_year("2023-05-15") == 2023

    def test_year_outside_range_returns_none(self):
        assert extract_year("2100") is None  # Too future
        assert extract_year("1234") is None  # Before 1500

    def test_invalid_year_returns_none(self):
        assert extract_year("forthcoming") is None


class TestAuthorNameParsing:
    def test_name_particles_van_der(self):
        result = parse_author_name("Johannes van der Waals")
        assert result.firstName == "Johannes"
        assert result.lastName == "van der Waals"

    def test_misplaced_initials_correction(self):
        result = parse_author_name("Benjamin A. Black")
        assert result.lastName == "Black"
        assert "A." in result.firstName


class TestMetadataMerge:
    def test_baseline_preferred_for_priority_fields(self):
        baseline = {"date": "2023-05-15"}
        extracted = {"date": "sometime in 2023"}
        result = merge_metadata_with_baseline(baseline, extracted)
        assert result["date"] == "2023-05-15"  # Baseline wins
```

## Prevention

When handling LLM-extracted metadata:

1. **Always validate years** before storing:
   ```python
   validated = validate_year(llm_output.get("year"))
   if validated:
       store_year(validated)
   ```

2. **Use structured author parsing** instead of simple string splitting:
   ```python
   # Bad: author_name.split(" ", 1)
   # Good: parse_author_name(author_name)
   ```

3. **Define merge strategy** for multiple data sources:
   ```python
   # Identify which sources are trusted for which fields
   baseline_priority_fields = {"date", "authors", "year"}
   ```

4. **Fail safely** - return None rather than storing invalid data

## Related Solutions

- [JSON String Array Edge Case](./json-string-array-edge-case.md) - LLM structured output validation
- [Query Generation Extraction Fixes](./query-generation-supervisor-extraction-fixes.md) - Structured output reliability

## Related Patterns

- [LLM Interaction Patterns](../../patterns/llm-interaction/) - Prompt engineering and output validation

## References

- [Pydantic Model Validators](https://docs.pydantic.dev/latest/concepts/validators/#model-validators)
- [OpenAlex API](https://docs.openalex.org/) - Trusted bibliographic data source
- [Zotero Creator Schema](https://www.zotero.org/support/dev/web_api/v3/types_and_fields) - firstName/lastName requirements
