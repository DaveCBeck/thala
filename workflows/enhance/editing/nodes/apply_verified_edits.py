"""Apply verified edits node for editing workflow."""

import logging
from typing import Any

from workflows.enhance.editing.document_model import DocumentModel

logger = logging.getLogger(__name__)


def validate_edit_uniqueness(document_text: str, find_string: str) -> tuple[bool, int]:
    """Check if find string appears exactly once in document.

    Returns:
        Tuple of (is_unique, occurrence_count)
    """
    count = document_text.count(find_string)
    return count == 1, count


async def apply_verified_edits_node(state: dict) -> dict[str, Any]:
    """Apply validated edits to the document.

    This node:
    1. Collects all pending edits from fact-check and reference-check
    2. Validates each edit (find string must be unique)
    3. Applies valid edits to the document
    4. Logs skipped edits at INFO level
    5. Returns updated document model and edit summaries
    """
    document_model = DocumentModel.from_dict(
        state.get("updated_document_model", state["document_model"])
    )
    pending_edits = state.get("pending_edits", [])
    unresolved_items = state.get("unresolved_items", [])

    if not pending_edits:
        logger.info("No verified edits to apply")
        return {
            "applied_edits": [],
            "skipped_edits": [],
            "verify_complete": True,
        }

    # Get current document text
    document_text = document_model.to_markdown()

    applied_edits = []
    skipped_edits = []

    # Sort edits by confidence (highest first)
    sorted_edits = sorted(pending_edits, key=lambda e: e.get("confidence", 0), reverse=True)

    for edit in sorted_edits:
        find_string = edit.get("find", "")
        replace_string = edit.get("replace", "")
        edit_type = edit.get("edit_type", "unknown")
        justification = edit.get("justification", "")
        confidence = edit.get("confidence", 0)

        # Skip empty or invalid edits
        if not find_string or len(find_string) < 20:
            logger.info(
                f"Skipping edit: find string too short ({len(find_string)} chars) - {justification}"
            )
            skipped_edits.append({
                **edit,
                "skip_reason": "find_string_too_short",
            })
            continue

        # Validate uniqueness
        is_unique, count = validate_edit_uniqueness(document_text, find_string)

        if not is_unique:
            if count == 0:
                logger.info(
                    f"Skipping edit: find string not found in document - "
                    f"type={edit_type}, confidence={confidence:.2f}"
                )
                skipped_edits.append({
                    **edit,
                    "skip_reason": "find_string_not_found",
                })
            else:
                logger.info(
                    f"Skipping edit: find string appears {count} times (must be unique) - "
                    f"type={edit_type}, confidence={confidence:.2f}"
                )
                skipped_edits.append({
                    **edit,
                    "skip_reason": f"find_string_not_unique_count_{count}",
                })
            continue

        # Apply the edit
        document_text = document_text.replace(find_string, replace_string, 1)
        applied_edits.append(edit)
        logger.debug(
            f"Applied {edit_type} edit: confidence={confidence:.2f}, {justification[:50]}..."
        )

    # Log summary
    logger.info(
        f"Applied {len(applied_edits)} edits, skipped {len(skipped_edits)} edits"
    )

    # Add skipped edits to unresolved items
    for edit in skipped_edits:
        unresolved_items.append({
            "source": "apply_verified_edits",
            "section_id": edit.get("position_hint", "unknown"),
            "issue": f"Edit skipped ({edit.get('skip_reason', 'unknown')}): {edit.get('justification', '')}",
        })

    # Log all unresolved items at INFO level
    if unresolved_items:
        logger.info(f"=== Verification Phase: {len(unresolved_items)} unresolved items ===")
        for item in unresolved_items:
            logger.info(
                f"  [{item.get('source', 'unknown')}] {item.get('section_id', 'unknown')}: "
                f"{item.get('issue', '')}"
            )

    # Rebuild document model from edited text
    # Note: This is a simplified approach - we're replacing the entire content
    # In a production system, you might want to rebuild the block structure more carefully
    updated_model = _rebuild_document_model_from_text(document_model, document_text)

    return {
        "updated_document_model": updated_model.to_dict(),
        "applied_edits": applied_edits,
        "skipped_edits": skipped_edits,
        "unresolved_items": unresolved_items,
        "verify_complete": True,
    }


def _rebuild_document_model_from_text(
    original_model: DocumentModel,
    edited_text: str,
) -> DocumentModel:
    """Rebuild document model from edited markdown text.

    This preserves the original structure as much as possible while
    incorporating the text changes from edits.
    """
    from workflows.enhance.editing.document_model import ContentBlock

    # For each section, find its content in the edited text
    # and update the blocks accordingly

    lines = edited_text.split("\n")
    current_section_id = None
    current_content_lines = []
    section_contents = {}

    for line in lines:
        # Check if this is a heading
        if line.startswith("#"):
            # Save previous section content
            if current_section_id:
                section_contents[current_section_id] = "\n".join(current_content_lines).strip()

            # Find matching section in original model
            heading_text = line.lstrip("#").strip()
            for section in original_model.get_all_sections():
                if section.heading == heading_text:
                    current_section_id = section.section_id
                    current_content_lines = []
                    break
            else:
                current_section_id = None
                current_content_lines = []
        elif current_section_id:
            current_content_lines.append(line)

    # Save last section
    if current_section_id:
        section_contents[current_section_id] = "\n".join(current_content_lines).strip()

    # Update each section's blocks
    for section in original_model.get_all_sections():
        if section.section_id in section_contents:
            new_content = section_contents[section.section_id]
            if new_content:
                # Split into paragraphs and create new blocks
                paragraphs = [p.strip() for p in new_content.split("\n\n") if p.strip()]
                new_blocks = []
                for para in paragraphs:
                    # Preserve original block types where possible
                    block = ContentBlock.from_content(para, "paragraph")
                    new_blocks.append(block)
                section.blocks = new_blocks

    return original_model
