"""Utility functions for supervision loops."""

from .paragraph_numbering import number_paragraphs, strip_paragraph_numbers
from .section_splitting import split_into_sections, SectionInfo
from .edit_application import validate_edits, apply_edits, EditValidationResult
from .structural_edit_application import (
    validate_structural_edits,
    apply_structural_edits,
    verify_edits_applied,
    StructuralEditValidationResult,
)
from .paper_formatting import (
    format_paper_summary_enhanced,
    format_paper_summaries_with_budget,
    create_manifest_note,
)
from .revision_history import document_revision
from .citation_validation import (
    extract_citation_keys_from_text,
    validate_edit_citations,
    check_section_growth,
)

__all__ = [
    "number_paragraphs",
    "strip_paragraph_numbers",
    "split_into_sections",
    "SectionInfo",
    "validate_edits",
    "apply_edits",
    "EditValidationResult",
    "validate_structural_edits",
    "apply_structural_edits",
    "verify_edits_applied",
    "StructuralEditValidationResult",
    "format_paper_summary_enhanced",
    "format_paper_summaries_with_budget",
    "create_manifest_note",
    "document_revision",
    "extract_citation_keys_from_text",
    "validate_edit_citations",
    "check_section_growth",
]
