"""Types for structural edit validation and application."""

from typing_extensions import TypedDict

from ...types import StructuralEdit


class StructuralEditValidationResult(TypedDict):
    """Result of structural edit validation."""

    valid_edits: list[StructuralEdit]
    invalid_edits: list[StructuralEdit]
    needs_retry_edits: list[StructuralEdit]  # Edits missing replacement_text
    errors: dict[int, str]  # Edit index -> error message
