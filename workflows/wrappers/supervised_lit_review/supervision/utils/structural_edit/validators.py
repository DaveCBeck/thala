"""Structural edit validation."""

import logging
from typing import Optional

from ...types import StructuralEdit
from .validation import StructuralEditValidationResult

logger = logging.getLogger(__name__)


def validate_structural_edits(
    paragraph_mapping: dict[int, str],
    edits: list[StructuralEdit],
) -> StructuralEditValidationResult:
    """Validate that structural edits reference valid paragraphs.

    Checks:
    - source_paragraph exists in mapping
    - target_paragraph exists in mapping (if required by edit_type)
    - source != target for reorder/merge
    - replacement_text exists for trim_redundancy and split_section

    Args:
        paragraph_mapping: {paragraph_num: paragraph_text} from number_paragraphs()
        edits: List of StructuralEdit objects to validate

    Returns:
        StructuralEditValidationResult with categorized edits and error messages.
        needs_retry_edits contains edits missing replacement_text that should trigger a retry.
    """
    valid_edits: list[StructuralEdit] = []
    invalid_edits: list[StructuralEdit] = []
    needs_retry_edits: list[StructuralEdit] = []
    errors: dict[int, str] = {}

    max_para = max(paragraph_mapping.keys()) if paragraph_mapping else 0

    for idx, edit in enumerate(edits):
        error: Optional[str] = None
        needs_retry = False

        # Check source exists
        if edit.source_paragraph not in paragraph_mapping:
            error = f"source_paragraph {edit.source_paragraph} does not exist (max: {max_para})"

        # Check target exists (if required)
        elif edit.edit_type in (
            "reorder_sections",
            "merge_sections",
            "add_transition",
            "move_content",
        ):
            if edit.target_paragraph is None:
                error = f"{edit.edit_type} requires target_paragraph"
            elif edit.target_paragraph not in paragraph_mapping:
                error = f"target_paragraph {edit.target_paragraph} does not exist (max: {max_para})"
            elif edit.source_paragraph == edit.target_paragraph:
                error = f"source and target are the same ({edit.source_paragraph})"

        # Check replacement_text for types that require it
        elif edit.edit_type == "trim_redundancy":
            if not edit.replacement_text:
                needs_retry = True
                error = f"trim_redundancy requires replacement_text (missing for P{edit.source_paragraph})"

        elif edit.edit_type == "split_section":
            if not edit.replacement_text:
                needs_retry = True
                error = (
                    "split_section requires replacement_text with ---SPLIT--- delimiter"
                )
            elif "---SPLIT---" not in edit.replacement_text:
                error = (
                    "split_section replacement_text must contain ---SPLIT--- delimiter"
                )

        elif edit.edit_type == "add_structural_content":
            if not edit.replacement_text:
                needs_retry = True
                error = "add_structural_content requires replacement_text with the structural content to add"

        if error:
            if needs_retry:
                needs_retry_edits.append(edit)
                logger.info(f"Edit {idx} needs retry: {error}")
            else:
                invalid_edits.append(edit)
                logger.warning(f"Invalid structural edit {idx}: {error}")
            errors[idx] = error
        else:
            valid_edits.append(edit)

    return StructuralEditValidationResult(
        valid_edits=valid_edits,
        invalid_edits=invalid_edits,
        needs_retry_edits=needs_retry_edits,
        errors=errors,
    )


def verify_edits_applied(
    original_mapping: dict[int, str],
    new_text: str,
    edits: list[StructuralEdit],
) -> dict[str, bool]:
    """Verify that edits were actually applied to the document.

    Checks for markers or content changes based on edit type.

    Args:
        original_mapping: Original paragraph mapping before edits
        new_text: Document text after edits
        edits: List of edits that should have been applied

    Returns:
        Dict of "{idx}_{edit_type}" -> was_applied for each edit
    """
    verifications: dict[str, bool] = {}

    for idx, edit in enumerate(edits):
        key = f"{idx}_{edit.edit_type}"

        if edit.edit_type == "reorder_sections":
            # Check that source paragraph content still exists
            source_text = original_mapping.get(edit.source_paragraph, "")[:50]
            verifications[key] = source_text in new_text if source_text else False

        elif edit.edit_type == "merge_sections":
            # Check both paragraphs' content appears
            src_text = original_mapping.get(edit.source_paragraph, "")[:30]
            tgt_text = original_mapping.get(edit.target_paragraph or 0, "")[:30]
            verifications[key] = (src_text in new_text if src_text else True) and (
                tgt_text in new_text if tgt_text else True
            )

        elif edit.edit_type == "add_transition":
            verifications[key] = "TRANSITION NEEDED" in new_text

        elif edit.edit_type == "delete_paragraph":
            # Check that source paragraph content is no longer present
            source_text = original_mapping.get(edit.source_paragraph, "")[:50]
            verifications[key] = source_text not in new_text if source_text else True

        elif edit.edit_type == "trim_redundancy":
            # Check that replacement_text is present (if provided)
            if edit.replacement_text:
                verifications[key] = edit.replacement_text[:50] in new_text
            else:
                # Fallback case - check for TODO marker
                verifications[key] = "TODO: Trim redundancy" in new_text

        elif edit.edit_type == "move_content":
            # Check that content appears at target (hard to verify precisely)
            # Just check target paragraph still exists
            tgt_text = original_mapping.get(edit.target_paragraph or 0, "")[:30]
            verifications[key] = tgt_text in new_text if tgt_text else False

        elif edit.edit_type == "split_section":
            # Check that ---SPLIT--- delimiter is not in output (was processed)
            # and content exists
            verifications[key] = "---SPLIT---" not in new_text

        elif edit.edit_type == "add_structural_content":
            # Check that replacement_text content is present in the new document
            if edit.replacement_text:
                # Check for a meaningful fragment (first 100 chars)
                content_fragment = edit.replacement_text[:100].strip()
                verifications[key] = content_fragment in new_text
            else:
                verifications[key] = False

    return verifications
