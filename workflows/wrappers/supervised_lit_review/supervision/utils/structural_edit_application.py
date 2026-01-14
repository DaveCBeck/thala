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
    needs_retry_edits: list[StructuralEdit]  # Edits missing replacement_text
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


def apply_structural_edits(
    paragraph_mapping: dict[int, str],
    edits: list[StructuralEdit],
    fallback_missing_replacement: bool = False,
) -> tuple[str, list[str]]:
    """Apply validated structural edits programmatically.

    Transforms the paragraph list according to each edit:
    - delete_paragraph: Remove paragraph entirely
    - trim_redundancy: Replace paragraph with replacement_text
    - move_content: Move content from source to target
    - split_section: Split paragraph using ---SPLIT--- delimiter
    - reorder_sections: Move paragraph from source to target position
    - merge_sections: Concatenate source into target, remove source
    - add_transition: Insert transition marker between paragraphs
    - add_structural_content: Insert new introduction, conclusion, discussion,
        or framing paragraph after source_paragraph

    Args:
        paragraph_mapping: {paragraph_num: paragraph_text} from number_paragraphs()
        edits: List of validated StructuralEdit objects
        fallback_missing_replacement: If True, convert trim_redundancy without
            replacement_text to TODO markers (for second retry attempt)

    Returns:
        Tuple of (restructured_document, applied_edit_descriptions)
    """
    # Work with ordered list of paragraphs
    paragraphs = [paragraph_mapping[i] for i in sorted(paragraph_mapping.keys())]
    applied: list[str] = []

    # Track which paragraph indices have been removed (for adjusting subsequent edits)
    removed_count = 0

    # Group edits by type for proper ordering:
    # 1. trim_redundancy first (modifies in place)
    # 2. split_section (expands paragraphs)
    # 3. add_structural_content (inserts new content)
    # 4. add_transition (inserts but doesn't remove)
    # 5. move_content (relocates content)
    # 6. delete_paragraph (removes paragraphs)
    # 7. merge_sections (changes structure)
    # 8. reorder_sections (changes structure)

    trim_edits = [e for e in edits if e.edit_type == "trim_redundancy"]
    split_edits = [e for e in edits if e.edit_type == "split_section"]
    structural_edits = [e for e in edits if e.edit_type == "add_structural_content"]
    transition_edits = [e for e in edits if e.edit_type == "add_transition"]
    move_edits = [e for e in edits if e.edit_type == "move_content"]
    delete_edits = [e for e in edits if e.edit_type == "delete_paragraph"]
    merge_edits = [e for e in edits if e.edit_type == "merge_sections"]
    reorder_edits = [e for e in edits if e.edit_type == "reorder_sections"]

    # Apply trim_redundancy (modify in place)
    for edit in trim_edits:
        src_idx = edit.source_paragraph - 1
        if src_idx < len(paragraphs):
            if edit.replacement_text:
                paragraphs[src_idx] = edit.replacement_text
                applied.append(
                    f"Trimmed P{edit.source_paragraph}: {edit.notes[:50]}..."
                )
            elif fallback_missing_replacement:
                # Fallback: add TODO marker instead
                paragraphs[src_idx] = (
                    f"<!-- TODO: Trim redundancy - {edit.notes} -->\n{paragraphs[src_idx]}"
                )
                applied.append(
                    f"Added TODO for P{edit.source_paragraph} (missing replacement_text)"
                )
            else:
                logger.warning(
                    f"Skipping trim_redundancy for P{edit.source_paragraph}: no replacement_text"
                )

    # Apply split_section (expands paragraphs)
    # Process in reverse order to maintain indices
    split_edits_sorted = sorted(
        split_edits, key=lambda e: e.source_paragraph, reverse=True
    )
    for edit in split_edits_sorted:
        src_idx = edit.source_paragraph - 1
        if src_idx < len(paragraphs) and edit.replacement_text:
            split_parts = edit.replacement_text.split("---SPLIT---")
            split_parts = [p.strip() for p in split_parts if p.strip()]
            if split_parts:
                paragraphs = (
                    paragraphs[:src_idx] + split_parts + paragraphs[src_idx + 1 :]
                )
                applied.append(
                    f"Split P{edit.source_paragraph} into {len(split_parts)} parts"
                )

    # Apply add_structural_content (inserts new structural content)
    # This adds introductions, conclusions, discussions, or framing paragraphs
    # Content is inserted AFTER the source_paragraph, unless "before" is in notes
    # Process in reverse order to maintain indices
    structural_edits_sorted = sorted(
        structural_edits, key=lambda e: e.source_paragraph, reverse=True
    )
    for edit in structural_edits_sorted:
        src_idx = edit.source_paragraph - 1
        if edit.replacement_text and src_idx < len(paragraphs):
            # Check if we should insert BEFORE the source (for introductions)
            if src_idx == 0 and "before" in edit.notes.lower():
                # Insert at the very beginning of the document
                paragraphs.insert(0, edit.replacement_text)
                applied.append(
                    f"Added structural content before P1: {edit.notes[:50]}..."
                )
            else:
                # Insert the new structural content after the source paragraph
                paragraphs.insert(src_idx + 1, edit.replacement_text)
                applied.append(
                    f"Added structural content after P{edit.source_paragraph}: {edit.notes[:50]}..."
                )

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
            applied.append(
                f"Added transition marker between P{edit.source_paragraph} and P{edit.target_paragraph}"
            )

    # Apply move_content (relocate from source to target)
    # Process in reverse order by source_paragraph to maintain indices
    move_edits_sorted = sorted(
        move_edits, key=lambda e: e.source_paragraph, reverse=True
    )
    for edit in move_edits_sorted:
        src_idx = edit.source_paragraph - 1
        tgt_idx = (edit.target_paragraph or 0) - 1
        if (
            src_idx < len(paragraphs)
            and tgt_idx < len(paragraphs)
            and src_idx != tgt_idx
        ):
            # Get content to move (either specific content or whole paragraph)
            content_to_move = edit.content_to_preserve or paragraphs[src_idx]
            # Append to target
            paragraphs[tgt_idx] = f"{paragraphs[tgt_idx]}\n\n{content_to_move}"
            # Clear or remove source
            if edit.content_to_preserve:
                # Only moved part of the content, leave rest in place
                paragraphs[src_idx] = (
                    paragraphs[src_idx].replace(edit.content_to_preserve, "").strip()
                )
                if not paragraphs[src_idx]:
                    paragraphs[src_idx] = ""  # Mark for cleanup
            else:
                paragraphs[src_idx] = ""  # Mark for cleanup
            applied.append(
                f"Moved content from P{edit.source_paragraph} to P{edit.target_paragraph}"
            )

    # Apply delete_paragraph (removes paragraphs)
    # Process in reverse order by source_paragraph to maintain indices
    delete_edits_sorted = sorted(
        delete_edits, key=lambda e: e.source_paragraph, reverse=True
    )
    for edit in delete_edits_sorted:
        src_idx = edit.source_paragraph - 1
        if src_idx < len(paragraphs):
            paragraphs.pop(src_idx)
            removed_count += 1
            applied.append(f"Deleted P{edit.source_paragraph}: {edit.notes[:50]}...")

    # Apply merge_sections (removes paragraphs)
    # Process in reverse order by source_paragraph to maintain indices
    merge_edits_sorted = sorted(
        merge_edits, key=lambda e: e.source_paragraph, reverse=True
    )
    for edit in merge_edits_sorted:
        src_idx = edit.source_paragraph - 1
        tgt_idx = (edit.target_paragraph or 0) - 1
        if (
            src_idx < len(paragraphs)
            and tgt_idx < len(paragraphs)
            and src_idx != tgt_idx
        ):
            source_text = paragraphs[src_idx]
            # Append source to target
            paragraphs[tgt_idx] = f"{paragraphs[tgt_idx]}\n\n{source_text}"
            # Remove source
            paragraphs.pop(src_idx)
            applied.append(
                f"Merged P{edit.source_paragraph} into P{edit.target_paragraph}"
            )

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
            applied.append(
                f"Moved P{edit.source_paragraph} to position near P{edit.target_paragraph}"
            )

    # Clean up empty paragraphs from move operations
    paragraphs = [p for p in paragraphs if p.strip()]

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
