"""Utility functions for supervision loops."""

from .paragraph_numbering import number_paragraphs, strip_paragraph_numbers
from .section_splitting import split_into_sections, SectionInfo
from .edit_application import validate_edits, apply_edits, EditValidationResult
from .revision_history import document_revision

__all__ = [
    "number_paragraphs",
    "strip_paragraph_numbers",
    "split_into_sections",
    "SectionInfo",
    "validate_edits",
    "apply_edits",
    "EditValidationResult",
    "document_revision",
]
