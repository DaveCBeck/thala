"""Architecture verification after edits are applied."""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.wrappers.supervised_lit_review.supervision.types import (
    ArchitectureVerificationResult,
)
from workflows.wrappers.supervised_lit_review.supervision.prompts import (
    LOOP3_VERIFIER_SYSTEM,
    LOOP3_VERIFIER_USER,
)

logger = logging.getLogger(__name__)


async def verify_architecture_node(state: dict) -> dict[str, Any]:
    """Verify that structural issues were resolved and document is coherent.

    This node runs after edits/rewrites are applied to confirm:
    1. Original issues are resolved
    2. No regressions introduced
    3. Document has coherent flow

    Works with both the new rewrite-based flow and legacy edit-based flow.
    """
    current_review = state["current_review"]
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]

    issue_analysis = state.get("issue_analysis", {})
    original_issues_text = issue_analysis.get("overall_assessment", "No assessment available")

    issues = issue_analysis.get("issues", [])
    if issues:
        issue_descriptions = [
            f"Issue {i.get('issue_id', '?')}: {i.get('issue_type', '?')} - {i.get('description', '')[:100]}"
            for i in issues
        ]
        original_issues_text += "\n\nSpecific issues identified:\n- " + "\n- ".join(issue_descriptions)

    rewrite_manifest = state.get("rewrite_manifest", {})
    changes_applied = state.get("changes_applied", [])

    if rewrite_manifest and rewrite_manifest.get("rewrites"):
        rewrites = rewrite_manifest.get("rewrites", [])
        applied_changes = [
            f"Rewrite for issue {r.get('issue_id', '?')}: {r.get('changes_summary', 'No summary')}"
            for r in rewrites
        ]
        applied_edits_text = "\n".join(f"- {c}" for c in applied_changes)
    elif changes_applied:
        applied_edits_text = "\n".join(f"- {c}" for c in changes_applied)
    else:
        applied_edits = state.get("applied_edits", [])
        applied_edits_text = "\n".join(f"- {e}" for e in applied_edits) if applied_edits else "None"

    user_prompt = LOOP3_VERIFIER_USER.format(
        original_issues=original_issues_text,
        applied_edits=applied_edits_text,
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
            f"Architecture verification complete: coherence={result.coherence_score:.2f}, "
            f"resolved={len(result.issues_resolved)}, remaining={len(result.issues_remaining)}, "
            f"regressions={len(result.regressions_introduced)}"
        )

        return {
            "architecture_verification": result.model_dump(),
            "needs_another_iteration": result.needs_another_iteration,
        }

    except Exception as e:
        logger.error(f"Architecture verification failed: {e}", exc_info=True)
        return {
            "architecture_verification": None,
            "needs_another_iteration": False,
        }
