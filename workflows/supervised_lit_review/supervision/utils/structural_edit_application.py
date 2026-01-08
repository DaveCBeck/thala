"""Structural edit validation and application for Loop 3."""

import logging
from typing import Optional

from typing_extensions import TypedDict

from ..types import StructuralEdit

logger = logging.getLogger(__name__)


class StructuralEditValidationResult(TypedDict):
    """Result of structural edit validation."""

    valid_edits: list[StructuralEdit]
    invalid_edits: list[StructuralEdit]
    errors: dict[int, str]  # Edit index -> error message


def validate_structural_edits(
    paragraph_mapping: dict[int, str],
    edits: list[StructuralEdit],
) -> StructuralEditValidationResult:
    """Validate that structural edits reference valid paragraphs.

    Checks:
    - source_paragraph exists in mapping
    - target_paragraph exists in mapping (if required by edit_type)
    - source != target for reorder/merge

    Args:
        paragraph_mapping: {paragraph_num: paragraph_text} from number_paragraphs()
        edits: List of StructuralEdit objects to validate

    Returns:
        StructuralEditValidationResult with categorized edits and error messages
    """
    valid_edits: list[StructuralEdit] = []
    invalid_edits: list[StructuralEdit] = []
    errors: dict[int, str] = {}

    max_para = max(paragraph_mapping.keys()) if paragraph_mapping else 0

    for idx, edit in enumerate(edits):
        error: Optional[str] = None

        # Check source exists
        if edit.source_paragraph not in paragraph_mapping:
            error = f"source_paragraph {edit.source_paragraph} does not exist (max: {max_para})"

        # Check target exists (if required)
        elif edit.edit_type in ("reorder_sections", "merge_sections", "add_transition"):
            if edit.target_paragraph is None:
                error = f"{edit.edit_type} requires target_paragraph"
            elif edit.target_paragraph not in paragraph_mapping:
                error = f"target_paragraph {edit.target_paragraph} does not exist (max: {max_para})"
            elif edit.source_paragraph == edit.target_paragraph:
                error = f"source and target are the same ({edit.source_paragraph})"

        if error:
            invalid_edits.append(edit)
            errors[idx] = error
            logger.warning(f"Invalid structural edit {idx}: {error}")
        else:
            valid_edits.append(edit)

    return StructuralEditValidationResult(
        valid_edits=valid_edits,
        invalid_edits=invalid_edits,
        errors=errors,
    )


def apply_structural_edits(
    paragraph_mapping: dict[int, str],
    edits: list[StructuralEdit],
) -> tuple[str, list[str]]:
    """Apply validated structural edits programmatically.

    Transforms the paragraph list according to each edit:
    - reorder_sections: Move paragraph from source to target position
    - merge_sections: Concatenate source into target, remove source
    - add_transition: Insert transition marker between paragraphs
    - flag_redundancy: Mark paragraph with redundancy comment

    Args:
        paragraph_mapping: {paragraph_num: paragraph_text} from number_paragraphs()
        edits: List of validated StructuralEdit objects

    Returns:
        Tuple of (restructured_document, applied_edit_descriptions)
    """
    # Work with ordered list of paragraphs
    paragraphs = [paragraph_mapping[i] for i in sorted(paragraph_mapping.keys())]
    applied: list[str] = []

    # Track which paragraph indices have been removed (for adjusting subsequent edits)
    # Process edits in order but adjust indices as we go
    removed_count = 0

    # Group edits by type for proper ordering:
    # 1. flag_redundancy first (doesn't change structure)
    # 2. add_transition second (inserts but doesn't remove)
    # 3. merge_sections (changes structure)
    # 4. reorder_sections (changes structure)

    flag_edits = [e for e in edits if e.edit_type == "flag_redundancy"]
    transition_edits = [e for e in edits if e.edit_type == "add_transition"]
    merge_edits = [e for e in edits if e.edit_type == "merge_sections"]
    reorder_edits = [e for e in edits if e.edit_type == "reorder_sections"]

    # Apply flag_redundancy (no structural change)
    for edit in flag_edits:
        src_idx = edit.source_paragraph - 1
        if src_idx < len(paragraphs):
            paragraphs[src_idx] = f"<!-- REDUNDANCY FLAG: {edit.notes} -->\n{paragraphs[src_idx]}"
            applied.append(f"Flagged P{edit.source_paragraph} as redundant: {edit.notes[:50]}...")

    # Apply add_transition (inserts marker, adjusts indices)
    # Sort by position, process in reverse to maintain indices
    transition_edits_sorted = sorted(
        transition_edits,
        key=lambda e: min(e.source_paragraph, e.target_paragraph or e.source_paragraph),
        reverse=True,
    )
    for edit in transition_edits_sorted:
        src_idx = edit.source_paragraph - 1
        tgt_idx = (edit.target_paragraph or edit.source_paragraph) - 1
        insert_pos = max(src_idx, tgt_idx)  # Insert after the later paragraph
        if insert_pos < len(paragraphs):
            transition = f"<!-- TRANSITION NEEDED between P{edit.source_paragraph} and P{edit.target_paragraph}: {edit.notes} -->"
            paragraphs.insert(insert_pos + 1, transition)
            applied.append(f"Added transition marker between P{edit.source_paragraph} and P{edit.target_paragraph}")

    # Apply merge_sections (removes paragraphs)
    # Process in reverse order by source_paragraph to maintain indices
    merge_edits_sorted = sorted(merge_edits, key=lambda e: e.source_paragraph, reverse=True)
    for edit in merge_edits_sorted:
        src_idx = edit.source_paragraph - 1
        tgt_idx = (edit.target_paragraph or 0) - 1
        if src_idx < len(paragraphs) and tgt_idx < len(paragraphs) and src_idx != tgt_idx:
            source_text = paragraphs[src_idx]
            # Append source to target
            paragraphs[tgt_idx] = f"{paragraphs[tgt_idx]}\n\n{source_text}"
            # Remove source
            paragraphs.pop(src_idx)
            applied.append(f"Merged P{edit.source_paragraph} into P{edit.target_paragraph}")

    # Apply reorder_sections
    # This is complex because each reorder affects subsequent indices
    # Process one at a time, recalculating indices each time
    for edit in reorder_edits:
        # Find current position of the source content
        # Note: After previous operations, we work with current positions
        src_idx = edit.source_paragraph - 1 - removed_count
        tgt_idx = (edit.target_paragraph or 0) - 1 - removed_count

        if 0 <= src_idx < len(paragraphs) and 0 <= tgt_idx < len(paragraphs):
            para = paragraphs.pop(src_idx)
            # Adjust target index if source was before target
            insert_idx = tgt_idx if src_idx > tgt_idx else tgt_idx - 1
            paragraphs.insert(insert_idx, para)
            applied.append(f"Moved P{edit.source_paragraph} to position near P{edit.target_paragraph}")

    # Reconstruct document
    restructured = "\n\n".join(paragraphs)

    return restructured, applied


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
            verifications[key] = (
                (src_text in new_text if src_text else True)
                and (tgt_text in new_text if tgt_text else True)
            )

        elif edit.edit_type == "add_transition":
            verifications[key] = "TRANSITION NEEDED" in new_text

        elif edit.edit_type == "flag_redundancy":
            verifications[key] = "REDUNDANCY FLAG" in new_text

    return verifications
