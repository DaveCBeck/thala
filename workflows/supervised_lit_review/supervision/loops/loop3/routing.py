"""Routing decisions for loop3 graph."""

import logging

logger = logging.getLogger(__name__)


def route_after_phase_a(state: dict) -> str:
    """Route based on Phase A results.

    Routes to:
    - generate_edits: If issues were identified and restructuring is needed
    - pass_through: If no issues found or restructuring not needed
    """
    analysis = state.get("issue_analysis")

    if not analysis:
        logger.info("No issue analysis, pass-through")
        return "pass_through"

    needs_restructuring = analysis.get("needs_restructuring", False)
    has_issues = bool(analysis.get("issues"))

    if needs_restructuring and has_issues:
        logger.info(f"Routing to Phase B ({len(analysis['issues'])} issues to resolve)")
        return "generate_edits"
    else:
        logger.info("No issues identified or restructuring not needed, pass-through")
        return "pass_through"


def route_after_analysis(state: dict) -> str:
    """Route based on whether restructuring is needed AND edits exist.

    Only routes to restructure if both:
    1. needs_restructuring is True
    2. At least one edit or todo_marker is provided

    This handles the case where the LLM sets needs_restructuring=True
    but fails to provide concrete edits.
    """
    manifest = state.get("edit_manifest")

    if not manifest:
        return "pass_through"

    needs_restructuring = manifest.get("needs_restructuring", False)
    has_edits = bool(manifest.get("edits")) or bool(manifest.get("todo_markers"))

    if needs_restructuring and has_edits:
        logger.info("Routing to validate_edits (restructuring needed with edits)")
        return "restructure_needed"
    elif needs_restructuring and not has_edits:
        logger.warning("needs_restructuring=True but no edits provided, treating as pass-through")
        return "pass_through"
    else:
        logger.info("Routing to finalize (pass-through)")
        return "pass_through"


def route_after_validation(state: dict) -> str:
    """Route based on validation results."""
    valid_edits = state.get("valid_edits", [])
    invalid_edits = state.get("invalid_edits", [])
    needs_retry_edits = state.get("needs_retry_edits", [])
    retry_attempted = state.get("retry_attempted", False)
    manifest = state.get("edit_manifest")

    if not manifest or not manifest.get("needs_restructuring"):
        return "no_edits"

    if needs_retry_edits and not retry_attempted:
        logger.info(f"{len(needs_retry_edits)} edits need retry for missing replacement_text")
        return "needs_retry"

    if valid_edits:
        logger.info(f"Routing to programmatic application ({len(valid_edits)} valid edits)")
        return "has_valid_edits"

    if invalid_edits or (needs_retry_edits and retry_attempted):
        logger.warning(
            f"Edits invalid or missing replacement_text after retry, falling back to LLM"
        )
        return "llm_fallback"

    return "no_edits"


def check_continue(state: dict) -> str:
    """Check if we should continue or complete the loop.

    Uses architecture verification results with STRICT coherence enforcement.
    Continues iteration if coherence < 0.8 AND issues remain.
    """
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]

    if iteration >= max_iterations - 1:
        logger.info(f"Max iterations reached ({max_iterations})")
        return "complete"

    arch_verification = state.get("architecture_verification")
    if arch_verification:
        needs_another = arch_verification.get("needs_another_iteration", False)
        coherence = arch_verification.get("coherence_score", 1.0)
        issues_remaining = len(arch_verification.get("issues_remaining", []))
        regressions = len(arch_verification.get("regressions_introduced", []))

        if coherence >= 0.8 and not needs_another:
            logger.info(f"Architecture verified (coherence={coherence:.2f}), completing")
            return "complete"

        if coherence < 0.8:
            if issues_remaining > 0 or regressions > 0:
                logger.info(
                    f"Coherence below threshold ({coherence:.2f}<0.8) with "
                    f"{issues_remaining} remaining issues, {regressions} regressions. "
                    "Continuing."
                )
                return "continue"
            elif needs_another:
                logger.info(f"Verifier requests another iteration (coherence={coherence:.2f})")
                return "continue"

        if needs_another:
            logger.info(f"Verifier requests another iteration (coherence={coherence:.2f})")
            return "continue"

    manifest = state.get("edit_manifest")
    issue_analysis = state.get("issue_analysis")

    if issue_analysis and issue_analysis.get("needs_restructuring"):
        issues_count = len(issue_analysis.get("issues", []))
        edits_count = len(manifest.get("edits", [])) if manifest else 0

        if issues_count > 0 and edits_count == 0:
            if iteration < max_iterations - 1:
                logger.warning(
                    f"Phase A identified {issues_count} issues but no edits generated. "
                    f"Continuing to iteration {iteration + 2}."
                )
                return "continue"
            else:
                logger.warning(
                    f"Max iterations reached with {issues_count} unresolved issues "
                    f"(no edits generated)."
                )
                return "complete"

    if manifest:
        edits_count = len(manifest.get("edits", []))
        if edits_count == 0:
            logger.info("No edits in manifest, completing")
            return "complete"

    logger.info(f"Continuing to iteration {iteration + 2}")
    return "continue"
