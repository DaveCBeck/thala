"""Structural edit application logic."""

import logging

from ...types import StructuralEdit

logger = logging.getLogger(__name__)


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
                applied.append(f"Trimmed P{edit.source_paragraph}: {edit.notes[:50]}...")
            elif fallback_missing_replacement:
                # Fallback: add TODO marker instead
                paragraphs[src_idx] = f"<!-- TODO: Trim redundancy - {edit.notes} -->\n{paragraphs[src_idx]}"
                applied.append(f"Added TODO for P{edit.source_paragraph} (missing replacement_text)")
            else:
                logger.warning(f"Skipping trim_redundancy for P{edit.source_paragraph}: no replacement_text")

    # Apply split_section (expands paragraphs)
    # Process in reverse order to maintain indices
    split_edits_sorted = sorted(split_edits, key=lambda e: e.source_paragraph, reverse=True)
    for edit in split_edits_sorted:
        src_idx = edit.source_paragraph - 1
        if src_idx < len(paragraphs) and edit.replacement_text:
            split_parts = edit.replacement_text.split("---SPLIT---")
            split_parts = [p.strip() for p in split_parts if p.strip()]
            if split_parts:
                paragraphs = paragraphs[:src_idx] + split_parts + paragraphs[src_idx + 1:]
                applied.append(f"Split P{edit.source_paragraph} into {len(split_parts)} parts")

    # Apply add_structural_content (inserts new structural content)
    # This adds introductions, conclusions, discussions, or framing paragraphs
    # Content is inserted AFTER the source_paragraph, unless "before" is in notes
    # Process in reverse order to maintain indices
    structural_edits_sorted = sorted(structural_edits, key=lambda e: e.source_paragraph, reverse=True)
    for edit in structural_edits_sorted:
        src_idx = edit.source_paragraph - 1
        if edit.replacement_text and src_idx < len(paragraphs):
            # Check if we should insert BEFORE the source (for introductions)
            if src_idx == 0 and "before" in edit.notes.lower():
                # Insert at the very beginning of the document
                paragraphs.insert(0, edit.replacement_text)
                applied.append(f"Added structural content before P1: {edit.notes[:50]}...")
            else:
                # Insert the new structural content after the source paragraph
                paragraphs.insert(src_idx + 1, edit.replacement_text)
                applied.append(f"Added structural content after P{edit.source_paragraph}: {edit.notes[:50]}...")

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

    # Apply move_content (relocate from source to target)
    # Process in reverse order by source_paragraph to maintain indices
    move_edits_sorted = sorted(move_edits, key=lambda e: e.source_paragraph, reverse=True)
    for edit in move_edits_sorted:
        src_idx = edit.source_paragraph - 1
        tgt_idx = (edit.target_paragraph or 0) - 1
        if src_idx < len(paragraphs) and tgt_idx < len(paragraphs) and src_idx != tgt_idx:
            # Get content to move (either specific content or whole paragraph)
            content_to_move = edit.content_to_preserve or paragraphs[src_idx]
            # Append to target
            paragraphs[tgt_idx] = f"{paragraphs[tgt_idx]}\n\n{content_to_move}"
            # Clear or remove source
            if edit.content_to_preserve:
                # Only moved part of the content, leave rest in place
                paragraphs[src_idx] = paragraphs[src_idx].replace(edit.content_to_preserve, "").strip()
                if not paragraphs[src_idx]:
                    paragraphs[src_idx] = ""  # Mark for cleanup
            else:
                paragraphs[src_idx] = ""  # Mark for cleanup
            applied.append(f"Moved content from P{edit.source_paragraph} to P{edit.target_paragraph}")

    # Apply delete_paragraph (removes paragraphs)
    # Process in reverse order by source_paragraph to maintain indices
    delete_edits_sorted = sorted(delete_edits, key=lambda e: e.source_paragraph, reverse=True)
    for edit in delete_edits_sorted:
        src_idx = edit.source_paragraph - 1
        if src_idx < len(paragraphs):
            paragraphs.pop(src_idx)
            removed_count += 1
            applied.append(f"Deleted P{edit.source_paragraph}: {edit.notes[:50]}...")

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

    # Clean up empty paragraphs from move operations
    paragraphs = [p for p in paragraphs if p.strip()]

    # Reconstruct document
    restructured = "\n\n".join(paragraphs)

    return restructured, applied
