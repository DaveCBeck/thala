"""Verify structure node for editing workflow."""

import logging
from typing import Any, Optional

from langsmith import traceable

from workflows.enhance.editing.document_model import DocumentModel
from workflows.enhance.editing.schemas import (
    StructureVerification,
    CoherenceComparisonResult,
)
from workflows.enhance.editing.prompts import (
    STRUCTURE_VERIFICATION_SYSTEM,
    STRUCTURE_VERIFICATION_USER,
    COHERENCE_COMPARISON_SYSTEM,
    COHERENCE_COMPARISON_USER,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)

# Threshold for coherence drop that triggers Sonnet comparison
REGRESSION_THRESHOLD = 0.05


def summarize_structure(doc_model: DocumentModel) -> str:
    """Create a brief structural summary for verification."""
    lines = [
        f"Title: {doc_model.title}",
        f"Total words: {doc_model.total_words}",
        f"Sections: {doc_model.section_count}",
        f"Blocks: {doc_model.block_count}",
        "",
        "Section outline:",
    ]

    for section in doc_model.get_all_sections():
        indent = "  " * (section.level - 1)
        lines.append(f"{indent}- {section.heading} ({section.total_words} words)")

    return "\n".join(lines)


async def _check_coherence_regression(
    original_model: DocumentModel,
    edited_model: DocumentModel,
    topic: str,
    edits_summary: str,
) -> Optional[CoherenceComparisonResult]:
    """Call Sonnet to compare original and edited documents for coherence regression.

    Args:
        original_model: The original document model (before edits)
        edited_model: The edited document model (after edits)
        topic: Document topic for context
        edits_summary: Summary of edits applied

    Returns:
        CoherenceComparisonResult if successful, None on failure
    """
    # Render both documents as markdown for comparison
    original_md = original_model.to_markdown()
    edited_md = edited_model.to_markdown()

    user_prompt = COHERENCE_COMPARISON_USER.format(
        topic=topic,
        original_document=original_md,
        edited_document=edited_md,
        edits_summary=edits_summary,
    )

    logger.info("Calling Sonnet to compare document versions for coherence regression")

    try:
        result = await get_structured_output(
            output_schema=CoherenceComparisonResult,
            user_prompt=user_prompt,
            system_prompt=COHERENCE_COMPARISON_SYSTEM,
            tier=ModelTier.SONNET,
            max_tokens=2000,
        )
        logger.info(
            f"Coherence comparison: preferred={result.preferred_version}, "
            f"confidence={result.confidence:.2f}, "
            f"original={result.original_score:.2f}, edited={result.edited_score:.2f}"
        )
        return result
    except Exception as e:
        logger.error(f"Coherence comparison failed: {e}", exc_info=True)
        return None


@traceable(run_type="chain", name="EditingVerifyStructure")
async def verify_structure_node(state: dict) -> dict[str, Any]:
    """Verify structural edits improved coherence.

    Compares before/after states to assess improvement.
    """
    original_model = DocumentModel.from_dict(state["document_model"])
    updated_model = DocumentModel.from_dict(state.get("updated_document_model", state["document_model"]))
    completed_edits = state.get("completed_edits", [])

    # Build edit summary
    successful_edits = [e for e in completed_edits if e.get("success")]
    edits_summary = "\n".join([
        f"- {e.get('edit_type')}: {e.get('operation', 'applied')}"
        for e in successful_edits[:10]  # Limit for prompt
    ])

    if not edits_summary:
        edits_summary = "No edits were applied."

    user_prompt = STRUCTURE_VERIFICATION_USER.format(
        original_structure=summarize_structure(original_model),
        updated_structure=summarize_structure(updated_model),
        edits_applied=edits_summary,
    )

    logger.info("Verifying structure after edits")

    try:
        verification = await get_structured_output(
            output_schema=StructureVerification,
            user_prompt=user_prompt,
            system_prompt=STRUCTURE_VERIFICATION_SYSTEM,
            tier=ModelTier.DEEPSEEK_V3,
            max_tokens=2000,
        )

        logger.info(
            f"Verification: coherence={verification.coherence_score:.2f}, "
            f"resolved={len(verification.issues_resolved)}, "
            f"remaining={len(verification.issues_remaining)}, "
            f"regressions={len(verification.regressions)}"
        )

        iteration = state.get("structure_iteration", 0)
        max_iterations = state.get("max_structure_iterations", 3)
        min_coherence = state.get("quality_settings", {}).get("min_coherence_threshold", 0.75)
        topic = state["input"]["topic"]

        # === Coherence Regression Detection ===
        baseline_coherence = state.get("baseline_coherence_score")
        retry_used = state.get("coherence_regression_retry_used", False)

        if baseline_coherence is not None:
            coherence_drop = baseline_coherence - verification.coherence_score

            if coherence_drop > REGRESSION_THRESHOLD:
                logger.warning(
                    f"Potential coherence regression detected: "
                    f"baseline={baseline_coherence:.2f}, "
                    f"current={verification.coherence_score:.2f}, "
                    f"drop={coherence_drop:.2f}"
                )

                # Call Sonnet to compare both documents
                comparison = await _check_coherence_regression(
                    original_model=original_model,
                    edited_model=updated_model,
                    topic=topic,
                    edits_summary=edits_summary,
                )

                if comparison and comparison.preferred_version == "original" and comparison.confidence >= 0.6:
                    # Confirmed regression - need to rollback
                    regressions_desc = ", ".join(comparison.key_regressions[:3]) if comparison.key_regressions else "general coherence loss"

                    if not retry_used:
                        # First regression: retry without incrementing iteration
                        logger.warning(
                            f"Coherence regression confirmed (confidence={comparison.confidence:.2f}). "
                            f"Rolling back and retrying. Regressions: {regressions_desc}"
                        )
                        return {
                            "updated_document_model": state["document_model"],  # Rollback to original
                            "structure_verification": verification.model_dump(),
                            "coherence_regression_retry_used": True,
                            "structure_iteration": iteration,  # Don't increment!
                            "needs_more_structure_work": True,  # Retry
                        }
                    else:
                        # Second regression: give up and proceed to polish
                        warning_msg = (
                            f"Coherence regression persisted after retry "
                            f"(confidence={comparison.confidence:.2f}). "
                            f"Rolling back to original document. "
                            f"Regressions: {regressions_desc}"
                        )
                        logger.warning(warning_msg)
                        return {
                            "updated_document_model": state["document_model"],  # Rollback to original
                            "structure_verification": verification.model_dump(),
                            "coherence_regression_detected": True,
                            "coherence_regression_warning": warning_msg,
                            "needs_more_structure_work": False,  # Give up, proceed to polish
                            "structure_iteration": iteration + 1,
                        }

        # === Normal flow (no regression or regression not confirmed) ===
        needs_more = (
            verification.coherence_score < min_coherence
            and verification.needs_another_iteration
            and iteration < max_iterations - 1
        )

        return {
            "structure_verification": verification.model_dump(),
            "needs_more_structure_work": needs_more,
            "structure_iteration": iteration + 1,
        }

    except Exception as e:
        logger.error(f"Structure verification failed: {e}", exc_info=True)
        return {
            "structure_verification": StructureVerification(
                coherence_score=0.7,
                issues_resolved=[],
                issues_remaining=[],
                regressions=[],
                structure_improved=False,
                flow_improved=False,
                completeness_improved=False,
                reasoning=f"Verification failed: {e}",
            ).model_dump(),
            "needs_more_structure_work": False,
            "structure_iteration": state.get("structure_iteration", 0) + 1,
            "errors": [{"node": "verify_structure", "error": str(e)}],
        }


def check_structure_complete(state: dict) -> str:
    """Determine whether to continue structure work or proceed to polish.

    IMPORTANT: This fixes the bug in Loop 3 v1 where iteration check
    happened before verification signals.

    Returns:
        "continue_structure" or "proceed_to_polish"
    """
    verification_data = state.get("structure_verification", {})
    iteration = state.get("structure_iteration", 1)
    max_iterations = state.get("max_structure_iterations", 3)

    if not verification_data:
        logger.warning("No verification data, proceeding to polish")
        return "proceed_to_polish"

    # === Coherence Regression Handling ===
    # If final regression detected (after retry), proceed to polish with original document
    if state.get("coherence_regression_detected"):
        logger.warning("Coherence regression confirmed after retry, proceeding to polish with original document")
        return "proceed_to_polish"

    # If retry is pending (regression_retry_used but needs_more_structure_work), loop back
    if state.get("coherence_regression_retry_used") and state.get("needs_more_structure_work"):
        logger.info("Retrying structure work after coherence regression rollback")
        return "continue_structure"

    verification = StructureVerification.model_validate(verification_data)

    # Check verification signals FIRST (the fix!)
    min_coherence = state.get("quality_settings", {}).get("min_coherence_threshold", 0.75)

    if verification.coherence_score >= min_coherence:
        if not verification.issues_remaining:
            logger.info(
                f"Structure complete: coherence={verification.coherence_score:.2f} >= {min_coherence}"
            )
            return "proceed_to_polish"

    # Check if we can iterate more
    if iteration < max_iterations:
        if verification.needs_another_iteration:
            logger.info(
                f"Structure iteration {iteration} → {iteration + 1} "
                f"(coherence={verification.coherence_score:.2f})"
            )
            return "continue_structure"

    # Max iterations reached
    if verification.issues_remaining:
        logger.warning(
            f"Max structure iterations ({max_iterations}) reached with "
            f"{len(verification.issues_remaining)} issues remaining"
        )

    return "proceed_to_polish"
