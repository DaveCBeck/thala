"""Plan edits node for editing workflow."""

import json
import logging
from typing import Any

from workflows.enhance.editing.document_model import DocumentModel
from workflows.enhance.editing.schemas import (
    StructuralAnalysis,
    EditPlan,
    GenerateIntroductionEdit,
    GenerateConclusionEdit,
    GenerateSynthesisEdit,
    GenerateTransitionEdit,
    SectionMoveEdit,
    SectionMergeEdit,
    ContentConsolidationEdit,
    DeleteRedundantEdit,
)
from workflows.enhance.editing.prompts import EDIT_PLANNING_SYSTEM, EDIT_PLANNING_USER
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


def validate_edit_references(edit: Any, doc_model: DocumentModel) -> bool:
    """Validate that all IDs in an edit exist in the document."""
    # Get all referenced section IDs
    section_refs = []
    block_refs = []

    if hasattr(edit, "source_section_id"):
        section_refs.append(edit.source_section_id)
    if hasattr(edit, "target_section_id") and edit.target_section_id:
        section_refs.append(edit.target_section_id)
    if hasattr(edit, "primary_section_id"):
        section_refs.append(edit.primary_section_id)
    if hasattr(edit, "secondary_section_id"):
        section_refs.append(edit.secondary_section_id)
    if hasattr(edit, "from_section_id"):
        section_refs.append(edit.from_section_id)
    if hasattr(edit, "to_section_id"):
        section_refs.append(edit.to_section_id)
    if hasattr(edit, "context_section_ids"):
        section_refs.extend(edit.context_section_ids)

    if hasattr(edit, "source_block_ids"):
        block_refs.extend(edit.source_block_ids)
    if hasattr(edit, "block_ids_to_delete"):
        block_refs.extend(edit.block_ids_to_delete)
    if hasattr(edit, "primary_block_id"):
        block_refs.append(edit.primary_block_id)
    if hasattr(edit, "block_id"):
        block_refs.append(edit.block_id)

    # Validate sections
    for sec_id in section_refs:
        if not doc_model.get_section(sec_id):
            logger.warning(f"Invalid section reference: {sec_id}")
            return False

    # Validate blocks
    for block_id in block_refs:
        if not doc_model.get_block(block_id):
            logger.warning(f"Invalid block reference: {block_id}")
            return False

    return True


def _find_last_content_section(all_sections: list) -> str | None:
    """Find the last content section (before References/Bibliography)."""
    if not all_sections:
        return None

    # Find References/Bibliography section if it exists
    references_idx = None
    for i, section in enumerate(all_sections):
        heading_lower = section.heading.lower()
        if any(kw in heading_lower for kw in ["reference", "bibliography", "works cited"]):
            references_idx = i
            break

    if references_idx is not None and references_idx > 0:
        # Return the section just before References
        return all_sections[references_idx - 1].section_id
    else:
        # No References section, use the last section
        return all_sections[-1].section_id


def create_edits_from_issues(
    analysis: StructuralAnalysis,
    doc_model: DocumentModel,
) -> list[Any]:
    """Create edit objects from structural issues.

    This is a heuristic-based approach that creates appropriate
    edits based on issue type and recommended action.
    """
    edits = []

    # Get all sections for context
    all_sections = doc_model.get_all_sections()
    first_section_id = all_sections[0].section_id if all_sections else None
    last_section_id = all_sections[-1].section_id if all_sections else None
    last_content_section_id = _find_last_content_section(all_sections)

    for issue in analysis.issues:
        try:
            if issue.recommended_action == "generate_intro":
                # Find context sections (first few sections for document intro)
                context_ids = [s.section_id for s in all_sections[:3]]
                edit = GenerateIntroductionEdit(
                    scope="document" if "document" in issue.description.lower() else "section",
                    target_section_id=issue.affected_section_ids[0] if issue.affected_section_ids else None,
                    context_section_ids=context_ids,
                    introduction_requirements=issue.action_details,
                    target_word_count=250,
                )
                edits.append(edit)

            elif issue.recommended_action == "generate_conclusion":
                # Context from last few content sections (excluding References)
                content_sections = [s for s in all_sections
                                   if not any(kw in s.heading.lower()
                                             for kw in ["reference", "bibliography"])]
                context_ids = [s.section_id for s in content_sections[-3:]]

                is_document_scope = "document" in issue.description.lower()
                edit = GenerateConclusionEdit(
                    scope="document" if is_document_scope else "section",
                    # For section scope, add to the specified section
                    target_section_id=issue.affected_section_ids[0] if issue.affected_section_ids and not is_document_scope else None,
                    # For document scope, create new section after last content section
                    insert_after_section_id=last_content_section_id if is_document_scope else None,
                    context_section_ids=context_ids,
                    conclusion_requirements=issue.action_details,
                    target_word_count=300,
                    new_section_heading="Conclusion" if is_document_scope else "Conclusion",
                )
                edits.append(edit)

            elif issue.recommended_action == "generate_synthesis":
                if issue.affected_section_ids:
                    edit = GenerateSynthesisEdit(
                        target_section_id=issue.affected_section_ids[0],
                        synthesis_requirements=issue.action_details,
                        position="end",
                        target_word_count=400,
                    )
                    edits.append(edit)

            elif issue.recommended_action == "generate_transition":
                if len(issue.affected_section_ids) >= 2:
                    edit = GenerateTransitionEdit(
                        from_section_id=issue.affected_section_ids[0],
                        to_section_id=issue.affected_section_ids[1],
                        transition_type="bridging",
                        target_word_count=100,
                    )
                    edits.append(edit)

            elif issue.recommended_action == "move_section":
                if len(issue.affected_section_ids) >= 2:
                    edit = SectionMoveEdit(
                        source_section_id=issue.affected_section_ids[0],
                        target_position="after",
                        target_section_id=issue.affected_section_ids[1],
                        justification=issue.action_details,
                    )
                    edits.append(edit)

            elif issue.recommended_action == "merge_sections":
                if len(issue.affected_section_ids) >= 2:
                    edit = SectionMergeEdit(
                        primary_section_id=issue.affected_section_ids[0],
                        secondary_section_id=issue.affected_section_ids[1],
                        merge_strategy="synthesize",
                        justification=issue.action_details,
                    )
                    edits.append(edit)

            elif issue.recommended_action == "consolidate":
                if issue.affected_block_ids and issue.affected_section_ids:
                    edit = ContentConsolidationEdit(
                        topic=issue.description[:50],
                        source_block_ids=issue.affected_block_ids[:5],  # Limit scope
                        target_section_id=issue.affected_section_ids[0],
                        consolidation_approach="synthesize",
                        justification=issue.action_details,
                    )
                    edits.append(edit)

            elif issue.recommended_action == "delete_redundant":
                if issue.affected_block_ids and len(issue.affected_block_ids) >= 2:
                    edit = DeleteRedundantEdit(
                        block_ids_to_delete=issue.affected_block_ids[1:],
                        primary_block_id=issue.affected_block_ids[0],
                        justification=issue.action_details,
                    )
                    edits.append(edit)

        except Exception as e:
            logger.warning(f"Failed to create edit for issue {issue.issue_id}: {e}")

    return edits


async def plan_edits_node(state: dict) -> dict[str, Any]:
    """Create ordered edit plan from structural analysis.

    Args:
        state: Current workflow state

    Returns:
        State update with edit_plan
    """
    analysis = StructuralAnalysis.model_validate(state["structural_analysis"])
    # Use updated document model if available (from previous iterations)
    document_model = DocumentModel.from_dict(
        state.get("updated_document_model", state["document_model"])
    )

    # If no structural work needed, skip
    if not analysis.needs_structural_work and not analysis.issues:
        logger.info("No structural work needed, creating empty edit plan")
        return {
            "edit_plan": EditPlan(
                edits=[],
                execution_order_rationale="No structural issues identified",
                estimated_word_count_change=0,
            ).model_dump(),
            "plan_complete": True,
        }

    # Create edits from issues
    edits = create_edits_from_issues(analysis, document_model)

    # Validate all references
    valid_edits = []
    for edit in edits:
        if validate_edit_references(edit, document_model):
            valid_edits.append(edit)
        else:
            logger.warning(f"Skipping edit with invalid references: {edit.edit_type}")

    # Deduplicate edits - only one edit per (edit_type, target) combination
    # This is a safety net in case the LLM generates duplicates despite prompting
    seen_keys = set()
    deduped_edits = []
    for edit in valid_edits:
        # Create a deduplication key based on edit type and target
        if edit.edit_type in ("generate_introduction", "generate_conclusion"):
            # For document-level edits, key by type + scope
            key = (edit.edit_type, getattr(edit, "scope", "document"))
        elif edit.edit_type in ("generate_synthesis", "generate_transition"):
            # Key by type + target section
            key = (edit.edit_type, getattr(edit, "target_section_id", None))
        elif edit.edit_type == "section_move":
            key = (edit.edit_type, edit.source_section_id)
        elif edit.edit_type == "section_merge":
            # Order-independent key for merges
            ids = tuple(sorted([edit.primary_section_id, edit.secondary_section_id]))
            key = (edit.edit_type, ids)
        elif edit.edit_type == "consolidate":
            key = (edit.edit_type, edit.target_section_id)
        elif edit.edit_type in ("delete_redundant", "trim_redundancy"):
            key = (edit.edit_type, getattr(edit, "primary_block_id", None) or getattr(edit, "block_id", None))
        else:
            key = (edit.edit_type, id(edit))  # Unique fallback

        if key not in seen_keys:
            seen_keys.add(key)
            deduped_edits.append(edit)
        else:
            logger.warning(f"Skipping duplicate edit: {edit.edit_type} with key {key}")

    valid_edits = deduped_edits

    # Order edits by priority
    # 1. Structure generation (intro/conclusion)
    # 2. Moves
    # 3. Consolidation
    # 4. Merges
    # 5. Removal
    # 6. Transitions

    def edit_priority(edit) -> int:
        priority_map = {
            "generate_introduction": 1,
            "generate_conclusion": 2,
            "section_move": 3,
            "consolidate": 4,
            "section_merge": 5,
            "delete_redundant": 6,
            "trim_redundancy": 7,
            "generate_synthesis": 8,
            "generate_transition": 9,
        }
        return priority_map.get(edit.edit_type, 10)

    valid_edits.sort(key=edit_priority)

    # Estimate word count change
    word_change = 0
    for edit in valid_edits:
        if hasattr(edit, "target_word_count"):
            word_change += edit.target_word_count
        elif edit.edit_type in ("delete_redundant", "trim_redundancy"):
            word_change -= 100  # Rough estimate

    edit_plan = EditPlan(
        edits=valid_edits,
        execution_order_rationale=(
            f"Created {len(valid_edits)} edits from {len(analysis.issues)} issues. "
            "Ordered by: structure generation → moves → consolidation → removal → transitions."
        ),
        estimated_word_count_change=word_change,
    )

    logger.info(
        f"Edit plan created: {len(edit_plan.structure_edits)} structure, "
        f"{len(edit_plan.generation_edits)} generation, "
        f"{len(edit_plan.removal_edits)} removal"
    )

    return {
        "edit_plan": edit_plan.model_dump(),
        "plan_complete": True,
    }
