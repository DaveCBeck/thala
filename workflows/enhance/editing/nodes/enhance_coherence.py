"""Enhance coherence review node for editing workflow."""

import logging
from typing import Any

from workflows.enhance.editing.document_model import DocumentModel
from workflows.enhance.editing.schemas import EnhanceCoherenceReview
from workflows.enhance.editing.prompts import (
    ENHANCE_COHERENCE_REVIEW_SYSTEM,
    ENHANCE_COHERENCE_REVIEW_USER,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


async def enhance_coherence_review_node(state: dict) -> dict[str, Any]:
    """Review coherence after an enhancement pass.

    Evaluates overall document coherence and identifies sections
    that may need additional enhancement in the next iteration.
    """
    document_model = DocumentModel.from_dict(
        state["updated_document_model"]
    )
    topic = state["input"]["topic"]
    iteration = state.get("enhance_iteration", 0)
    max_iterations = state.get("max_enhance_iterations", 3)
    enhancements = state.get("section_enhancements", [])

    # Get sections that were enhanced in this pass
    enhanced_section_ids = [
        e["section_id"] for e in enhancements
        if e.get("success")
    ]

    # Render document for review
    document_text = document_model.to_markdown()

    logger.info(
        f"Reviewing coherence after enhancement pass {iteration + 1}/{max_iterations} "
        f"({len(enhanced_section_ids)} sections enhanced)"
    )

    user_prompt = ENHANCE_COHERENCE_REVIEW_USER.format(
        topic=topic,
        iteration=iteration + 1,
        max_iterations=max_iterations,
        sections_enhanced=", ".join(enhanced_section_ids) or "none",
        document_text=document_text,  # Pass full document for accurate review
    )

    try:
        result = await get_structured_output(
            output_schema=EnhanceCoherenceReview,
            user_prompt=user_prompt,
            system_prompt=ENHANCE_COHERENCE_REVIEW_SYSTEM,
            tier=ModelTier.SONNET,
            max_tokens=2000,
            use_json_schema_method=True,
        )

        logger.info(
            f"Coherence review: score={result.coherence_score:.2f}, "
            f"needs_work={len(result.sections_needing_work)}, "
            f"issues={len(result.issues_found)}"
        )

        return {
            "enhance_coherence_review": result.model_dump(),
            "enhance_flagged_sections": result.sections_needing_work,
            "enhance_iteration": iteration + 1,
        }

    except Exception as e:
        logger.error(f"Coherence review failed: {e}", exc_info=True)
        # On failure, don't flag any sections (stop iterating)
        return {
            "enhance_coherence_review": EnhanceCoherenceReview(
                coherence_score=0.7,
                sections_enhanced=enhanced_section_ids,
                sections_needing_work=[],
                issues_found=[f"Review failed: {e}"],
                overall_assessment="Coherence review could not be completed.",
            ).model_dump(),
            "enhance_flagged_sections": [],
            "enhance_iteration": iteration + 1,
            "errors": [{"node": "enhance_coherence_review", "error": str(e)}],
        }


def route_enhance_iteration(state: dict) -> str:
    """Determine whether to continue enhancement or proceed to verification.

    Returns:
        "continue" to enhance more sections, "verify" to proceed to fact-checking
    """
    iteration = state.get("enhance_iteration", 0)
    max_iterations = state.get("max_enhance_iterations", 3)
    coherence_review = state.get("enhance_coherence_review", {})
    flagged_sections = state.get("enhance_flagged_sections", [])

    # Check coherence threshold
    min_coherence = state.get("quality_settings", {}).get("min_coherence_threshold", 0.75)
    coherence_score = coherence_review.get("coherence_score", 0.7)

    # If we've reached max iterations, stop
    if iteration >= max_iterations:
        logger.info(
            f"Max enhancement iterations ({max_iterations}) reached. "
            f"Proceeding to verification."
        )
        return "verify"

    # If coherence is good enough and no sections flagged, stop
    if coherence_score >= min_coherence and not flagged_sections:
        logger.info(
            f"Enhancement complete: coherence={coherence_score:.2f} >= {min_coherence}, "
            f"no sections flagged. Proceeding to verification."
        )
        return "verify"

    # If there are flagged sections and we can iterate, continue
    if flagged_sections:
        logger.info(
            f"Enhancement iteration {iteration} → {iteration + 1}: "
            f"{len(flagged_sections)} sections flagged for re-enhancement"
        )
        return "continue"

    # Default: proceed to verification
    return "verify"
