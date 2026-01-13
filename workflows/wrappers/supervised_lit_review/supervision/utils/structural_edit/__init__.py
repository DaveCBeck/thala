"""Structural edit validation and application for Loop 3."""

from .applier import apply_structural_edits
from .validation import StructuralEditValidationResult
from .validators import validate_structural_edits, verify_edits_applied

__all__ = [
    "apply_structural_edits",
    "validate_structural_edits",
    "verify_edits_applied",
    "StructuralEditValidationResult",
]
