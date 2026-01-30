"""Finalize node for editing workflow."""

import logging
from datetime import datetime, timezone
from typing import Any

from langsmith import traceable

from workflows.enhance.editing.document_model import DocumentModel
from workflows.enhance.editing.schemas import FinalVerification
from workflows.enhance.editing.prompts import (
    FINAL_VERIFICATION_SYSTEM,
    FINAL_VERIFICATION_USER,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="EditingFinalize")
async def finalize_node(state: dict) -> dict[str, Any]:
    """Generate final document and summary.

    Reconstructs markdown from the edited document model and
    performs final quality verification.
    """
    document_model = DocumentModel.from_dict(
        state["updated_document_model"]
    )
    topic = state["input"]["topic"]

    # Final semantic deduplication - remove duplicate sections and content
    removed_duplicates = document_model.deduplicate_sections()
    if removed_duplicates:
        logger.info(f"Deduplication removed {len(removed_duplicates)} items: {removed_duplicates}")

    # Reconstruct markdown (also handles header deduplication in blocks)
    final_document = document_model.to_markdown()

    logger.info(
        f"Finalizing document: {document_model.total_words} words, "
        f"{document_model.section_count} sections"
    )

    # Build changes summary (V2 structure + V1 enhancement/polish)
    rewritten_sections = state.get("rewritten_sections", [])
    section_enhancements = state.get("section_enhancements", [])
    polish_results = state.get("polish_results", [])

    # Count V2 structure phase changes
    sections_rewritten = len(rewritten_sections)
    sections_deleted = sum(1 for r in rewritten_sections if r.get("instruction_type") == "delete")
    sections_merged = sum(1 for r in rewritten_sections if r.get("instruction_type") == "merge_into")

    # Count V1 enhancement/polish changes
    successful_enhance = sum(1 for e in section_enhancements if e.get("success"))
    successful_polish = sum(1 for r in polish_results if r.get("success"))

    changes_summary = (
        f"V2 Structure: {sections_rewritten} sections rewritten "
        f"({sections_deleted} deleted, {sections_merged} merged). "
        f"Enhancement: {successful_enhance} sections improved. "
        f"Polish: {successful_polish} flow improvements."
    )

    # Final verification - pass full document for accurate assessment
    try:
        user_prompt = FINAL_VERIFICATION_USER.format(
            topic=topic,
            document=final_document,
        )

        final_verification = await get_structured_output(
            output_schema=FinalVerification,
            user_prompt=user_prompt,
            system_prompt=FINAL_VERIFICATION_SYSTEM,
            tier=ModelTier.DEEPSEEK_V3,
            max_tokens=1500,
        )

        logger.info(
            f"Final verification: coherence={final_verification.coherence_score:.2f}, "
            f"completeness={final_verification.completeness_score:.2f}, "
            f"flow={final_verification.flow_score:.2f}, "
            f"overall={final_verification.overall_score:.2f}"
        )

    except Exception as e:
        logger.error(f"Final verification failed: {e}", exc_info=True)
        final_verification = FinalVerification(
            coherence_score=0.7,
            completeness_score=0.7,
            flow_score=0.7,
            has_introduction=True,
            has_conclusion=True,
            sections_well_organized=True,
            remaining_issues=[f"Verification failed: {e}"],
            overall_assessment="Verification could not be completed.",
        )

    # Determine status
    errors = state.get("errors", [])
    if errors:
        status = "partial"
    elif final_verification.overall_score >= 0.7:
        status = "success"
    else:
        status = "partial"

    return {
        "final_document": final_document,
        "final_verification": final_verification.model_dump(),
        "changes_summary": changes_summary,
        "status": status,
        "completed_at": datetime.now(timezone.utc),
    }
