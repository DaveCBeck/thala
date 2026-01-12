"""Architecture verification after edits are applied."""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.supervised_lit_review.supervision.types import (
    ArchitectureVerificationResult,
)
from workflows.supervised_lit_review.supervision.prompts import (
    LOOP3_VERIFIER_SYSTEM,
    LOOP3_VERIFIER_USER,
)

logger = logging.getLogger(__name__)


async def verify_architecture_node(state: dict) -> dict[str, Any]:
    """Verify that structural issues were resolved and document is coherent.

    This node runs after edits are applied to confirm:
    1. Original issues are resolved
    2. No regressions introduced
    3. Document has coherent flow
    """
    current_review = state["current_review"]
    edit_manifest = state.get("edit_manifest", {})
    applied_edits = state.get("applied_edits", [])
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]

    original_issues = edit_manifest.get("overall_assessment", "No assessment available") if edit_manifest else "No manifest"
    architecture = edit_manifest.get("architecture_assessment", {}) if edit_manifest else {}
    if architecture:
        issues_list = (
            architecture.get("content_placement_issues", []) +
            architecture.get("logical_flow_issues", []) +
            architecture.get("anti_patterns_detected", [])
        )
        if issues_list:
            original_issues += "\n- " + "\n- ".join(issues_list)

    user_prompt = LOOP3_VERIFIER_USER.format(
        original_issues=original_issues,
        applied_edits="\n".join(f"- {e}" for e in applied_edits) if applied_edits else "None",
        current_document=current_review[:15000],
        iteration=iteration + 1,
        max_iterations=max_iterations,
    )

    try:
        result = await get_structured_output(
            output_schema=ArchitectureVerificationResult,
            user_prompt=user_prompt,
            system_prompt=LOOP3_VERIFIER_SYSTEM,
            tier=ModelTier.SONNET,
            thinking_budget=4000,
            max_tokens=4096,
            use_json_schema_method=True,
            max_retries=2,
        )

        logger.info(
            f"Architecture verification: coherence={result.coherence_score:.2f}, "
            f"resolved={len(result.issues_resolved)}, remaining={len(result.issues_remaining)}, "
            f"regressions={len(result.regressions_introduced)}"
        )

        return {
            "architecture_verification": result.model_dump(),
            "needs_another_iteration": result.needs_another_iteration,
        }

    except Exception as e:
        logger.error(f"Architecture verification failed: {e}")
        return {
            "architecture_verification": None,
            "needs_another_iteration": False,
        }
