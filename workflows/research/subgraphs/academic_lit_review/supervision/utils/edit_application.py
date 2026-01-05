"""Edit validation and application for Loop 5 fact/reference checking."""

from typing_extensions import TypedDict

from ..types import Edit


class EditValidationResult(TypedDict):
    """Result of edit validation."""

    valid_edits: list[Edit]
    invalid_edits: list[Edit]
    errors: dict[str, str]  # Edit index -> error message


def validate_edits(doc: str, edits: list[Edit]) -> EditValidationResult:
    """Validate that each edit's find string exists exactly once.

    Checks:
    - Find string must exist in document
    - Find string must be unambiguous (only one occurrence)

    Args:
        doc: Document text to validate against
        edits: List of Edit objects to validate

    Returns:
        EditValidationResult with categorized edits and error messages
    """
    valid_edits = []
    invalid_edits = []
    errors = {}

    for idx, edit in enumerate(edits):
        find_text = edit.find
        occurrences = doc.count(find_text)

        if occurrences == 0:
            invalid_edits.append(edit)
            errors[str(idx)] = "not_found"
        elif occurrences > 1:
            invalid_edits.append(edit)
            errors[str(idx)] = "ambiguous"
        else:
            valid_edits.append(edit)

    return EditValidationResult(
        valid_edits=valid_edits,
        invalid_edits=invalid_edits,
        errors=errors,
    )


def apply_edits(doc: str, edits: list[Edit]) -> str:
    """Apply validated edits to document.

    Assumes edits have been validated (each find string exists exactly once).
    Applies edits sequentially using str.replace.

    Args:
        doc: Document text to edit
        edits: List of validated Edit objects

    Returns:
        Edited document text
    """
    result = doc

    for edit in edits:
        result = result.replace(edit.find, edit.replace, 1)

    return result
