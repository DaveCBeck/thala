"""Loop 3: Structure and Cohesion - Structural editing with two-agent pattern."""

import logging
from typing import Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.academic_lit_review.state import LitReviewInput
from workflows.supervised_lit_review.supervision.types import (
    EditManifest,
    StructuralEdit,
    ArchitectureVerificationResult,
)
from workflows.supervised_lit_review.supervision.prompts import (
    LOOP3_ANALYST_SYSTEM,
    LOOP3_ANALYST_USER,
    LOOP3_EDITOR_SYSTEM,
    LOOP3_EDITOR_USER,
    LOOP3_VERIFIER_SYSTEM,
    LOOP3_VERIFIER_USER,
)
from workflows.supervised_lit_review.supervision.utils import (
    number_paragraphs,
    strip_paragraph_numbers,
    validate_structural_edits,
    apply_structural_edits,
    verify_edits_applied,
)

logger = logging.getLogger(__name__)


class Loop3State(TypedDict):
    """State for Loop 3 structural editing."""

    current_review: str
    numbered_document: str
    paragraph_mapping: dict[int, str]
    edit_manifest: Optional[dict]
    input: LitReviewInput
    iteration: int
    max_iterations: int
    is_complete: bool

    valid_edits: list[dict]
    invalid_edits: list[dict]
    needs_retry_edits: list[dict]  # Edits missing replacement_text
    validation_errors: dict[int, str]
    applied_edits: list[str]
    fallback_used: bool
    retry_attempted: bool  # Whether we already retried for missing replacement_text

    # Architecture verification
    architecture_verification: Optional[dict]
    needs_another_iteration: bool


def number_paragraphs_node(state: Loop3State) -> dict[str, Any]:
    """Add paragraph numbers to document for structural editing."""
    current_review = state["current_review"]

    if not current_review:
        logger.warning("No review content to number")
        return {
            "is_complete": True,
            "numbered_document": "",
            "paragraph_mapping": {},
        }

    numbered_doc, para_mapping = number_paragraphs(current_review)

    logger.info(f"Numbered document into {len(para_mapping)} paragraphs")

    return {
        "numbered_document": numbered_doc,
        "paragraph_mapping": para_mapping,
    }


async def analyze_structure_node(state: Loop3State) -> dict[str, Any]:
    """Analyze document structure and produce edit manifest.

    Uses Opus with extended thinking to carefully assess structural issues.
    Only suggests changes if they genuinely strengthen the piece.
    """
    numbered_doc = state["numbered_document"]
    input_data = state["input"]
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]

    topic = input_data.get("topic", "")

    user_prompt = LOOP3_ANALYST_USER.format(
        numbered_document=numbered_doc,
        topic=topic,
        iteration=iteration + 1,
        max_iterations=max_iterations,
    )

    try:
        manifest = await get_structured_output(
            output_schema=EditManifest,
            user_prompt=user_prompt,
            system_prompt=LOOP3_ANALYST_SYSTEM,
            tier=ModelTier.OPUS,
            thinking_budget=8000,
            max_tokens=12096,
            use_json_schema_method=True,
            max_retries=2,
        )

        if manifest.needs_restructuring and len(manifest.edits) == 0 and len(manifest.todo_markers) == 0:
            logger.warning(
                "Manifest indicates needs_restructuring=True but edits and todo_markers are empty. "
                f"Assessment: {manifest.overall_assessment[:100]}..."
            )

        logger.info(
            f"Loop 3 Analyst: needs_restructuring={manifest.needs_restructuring}, "
            f"edits={len(manifest.edits)}, todos={len(manifest.todo_markers)}"
        )

        return {"edit_manifest": manifest.model_dump()}

    except Exception as e:
        logger.error(f"Structure analysis failed: {e}")
        return {
            "edit_manifest": {
                "edits": [],
                "todo_markers": [],
                "overall_assessment": f"Analysis failed: {e}",
                "needs_restructuring": False,
            },
        }


def route_after_analysis(state: Loop3State) -> str:
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


def validate_edits_node(state: Loop3State) -> dict[str, Any]:
    """Validate that structural edits reference valid paragraphs."""
    manifest = state.get("edit_manifest")
    paragraph_mapping = state.get("paragraph_mapping", {})

    if not manifest or not manifest.get("edits"):
        logger.info("No edits to validate")
        return {
            "valid_edits": [],
            "invalid_edits": [],
            "needs_retry_edits": [],
            "validation_errors": {},
        }

    edits = [StructuralEdit(**e) for e in manifest.get("edits", [])]

    result = validate_structural_edits(paragraph_mapping, edits)

    logger.info(
        f"Edit validation: {len(result['valid_edits'])} valid, "
        f"{len(result['invalid_edits'])} invalid, "
        f"{len(result['needs_retry_edits'])} need retry"
    )

    if result["invalid_edits"]:
        for idx, error in result["errors"].items():
            logger.warning(f"Invalid edit {idx}: {error}")

    if result["needs_retry_edits"]:
        for edit in result["needs_retry_edits"]:
            logger.info(f"Edit needs retry (missing replacement_text): P{edit.source_paragraph}")

    return {
        "valid_edits": [e.model_dump() for e in result["valid_edits"]],
        "invalid_edits": [e.model_dump() for e in result["invalid_edits"]],
        "needs_retry_edits": [e.model_dump() for e in result["needs_retry_edits"]],
        "validation_errors": result["errors"],
    }


def route_after_validation(state: Loop3State) -> str:
    """Route based on validation results."""
    valid_edits = state.get("valid_edits", [])
    invalid_edits = state.get("invalid_edits", [])
    needs_retry_edits = state.get("needs_retry_edits", [])
    retry_attempted = state.get("retry_attempted", False)
    manifest = state.get("edit_manifest")

    if not manifest or not manifest.get("needs_restructuring"):
        return "no_edits"

    # If we have edits that need retry (missing replacement_text) and haven't tried yet,
    # route to retry. If we already retried, treat them as valid with fallback.
    if needs_retry_edits and not retry_attempted:
        logger.info(f"{len(needs_retry_edits)} edits need retry for missing replacement_text")
        return "needs_retry"

    if valid_edits:
        logger.info(f"Routing to programmatic application ({len(valid_edits)} valid edits)")
        return "has_valid_edits"

    if invalid_edits or (needs_retry_edits and retry_attempted):
        # If we retried but still missing replacement_text, use fallback mode
        logger.warning(
            f"Edits invalid or missing replacement_text after retry, falling back to LLM"
        )
        return "llm_fallback"

    return "no_edits"


def apply_edits_programmatically_node(state: Loop3State) -> dict[str, Any]:
    """Apply validated edits programmatically using paragraph mapping."""
    paragraph_mapping = state.get("paragraph_mapping", {})
    valid_edits = state.get("valid_edits", [])

    if not valid_edits:
        logger.info("No valid edits to apply programmatically")
        return {"fallback_used": False}

    edits = [StructuralEdit(**e) for e in valid_edits]

    restructured, applied_descriptions = apply_structural_edits(
        paragraph_mapping, edits
    )

    logger.info(f"Programmatically applied {len(applied_descriptions)} edits")
    for desc in applied_descriptions:
        logger.debug(f"  - {desc}")

    return {
        "current_review": restructured,
        "applied_edits": applied_descriptions,
        "fallback_used": False,
    }


def verify_application_node(state: Loop3State) -> dict[str, Any]:
    """Verify that edits were actually applied to the document."""
    original_mapping = state.get("paragraph_mapping", {})
    current_review = state.get("current_review", "")
    valid_edits = state.get("valid_edits", [])

    if not valid_edits:
        return {}

    edits = [StructuralEdit(**e) for e in valid_edits]
    verifications = verify_edits_applied(original_mapping, current_review, edits)

    failed = [k for k, v in verifications.items() if not v]
    if failed:
        logger.warning(f"Edit verification failures: {failed}")
    else:
        logger.info(f"All {len(verifications)} edits verified as applied")

    return {}


async def retry_analyze_node(state: Loop3State) -> dict[str, Any]:
    """Retry analysis specifically for edits missing replacement_text.

    Re-prompts the LLM with explicit instructions to provide replacement_text
    for trim_redundancy and split_section edits.
    """
    needs_retry_edits = state.get("needs_retry_edits", [])
    numbered_doc = state["numbered_document"]
    input_data = state["input"]
    topic = input_data.get("topic", "")

    # Build specific retry prompt
    retry_prompt = f"""The previous analysis identified edits that require replacement_text but it was missing.

Please provide the replacement_text for these specific edits:

"""
    for edit in needs_retry_edits:
        edit_type = edit.get("edit_type")
        source = edit.get("source_paragraph")
        notes = edit.get("notes", "")
        retry_prompt += f"""
- {edit_type} for P{source}: {notes}
  Please provide the replacement_text (the trimmed/split content).
"""

    retry_prompt += f"""
## Numbered Document
{numbered_doc}

## Research Topic
{topic}

Provide an EditManifest with the SAME edits but include replacement_text for each."""

    try:
        manifest = await get_structured_output(
            output_schema=EditManifest,
            user_prompt=retry_prompt,
            system_prompt=LOOP3_ANALYST_SYSTEM,
            tier=ModelTier.OPUS,
            thinking_budget=4000,
            max_tokens=8096,
            use_json_schema_method=True,
            max_retries=1,
        )

        logger.info(f"Retry analysis: {len(manifest.edits)} edits returned")

        return {
            "edit_manifest": manifest.model_dump(),
            "retry_attempted": True,
        }

    except Exception as e:
        logger.error(f"Retry analysis failed: {e}")
        return {"retry_attempted": True}


async def verify_architecture_node(state: Loop3State) -> dict[str, Any]:
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

    # Format original issues from manifest
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
        current_document=current_review[:15000],  # Limit size
        iteration=iteration + 1,
        max_iterations=max_iterations,
    )

    try:
        result = await get_structured_output(
            output_schema=ArchitectureVerificationResult,
            user_prompt=user_prompt,
            system_prompt=LOOP3_VERIFIER_SYSTEM,
            tier=ModelTier.SONNET,  # Sonnet is sufficient for verification
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


async def execute_manifest_node(state: Loop3State) -> dict[str, Any]:
    """Execute the edit manifest to restructure the document (LLM fallback).

    Used when programmatic edit application fails or isn't possible.
    Uses Opus to carefully execute structural changes while preserving
    citations and academic formatting.
    """
    from workflows.shared.llm_utils import get_llm

    numbered_doc = state["numbered_document"]
    manifest = state.get("edit_manifest")

    if not manifest:
        logger.warning("No manifest to execute")
        return {"fallback_used": True}

    manifest_text = _format_manifest(manifest)

    user_prompt = LOOP3_EDITOR_USER.format(
        numbered_document=numbered_doc,
        edit_manifest=manifest_text,
    )

    llm = get_llm(
        tier=ModelTier.OPUS,
        max_tokens=16384,
    )

    try:
        messages = [
            {"role": "system", "content": LOOP3_EDITOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        response = await llm.ainvoke(messages)

        restructured_text = ""
        if isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    restructured_text = block.get("text", "")
                    break
                elif hasattr(block, "type") and block.type == "text":
                    restructured_text = getattr(block, "text", "")
                    break
        else:
            restructured_text = response.content

        logger.info("Successfully executed edit manifest (LLM fallback)")

        return {
            "current_review": restructured_text,
            "fallback_used": True,
        }

    except Exception as e:
        logger.error(f"Manifest execution failed: {e}")
        return {"fallback_used": True}


def validate_result_node(state: Loop3State) -> dict[str, Any]:
    """Strip paragraph numbers and validate the restructured output."""
    current_review = state["current_review"]

    cleaned_review = strip_paragraph_numbers(current_review)

    if not cleaned_review or len(cleaned_review) < 100:
        logger.warning("Validation failed: output too short or empty")
        original_numbered = state.get("numbered_document", "")
        if original_numbered:
            cleaned_review = strip_paragraph_numbers(original_numbered)

    logger.info(f"Validated result: {len(cleaned_review)} characters")

    return {
        "current_review": cleaned_review,
    }


def check_continue(state: Loop3State) -> str:
    """Check if we should continue or complete the loop.

    Now uses architecture verification results to inform decision.
    """
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]

    if iteration >= max_iterations - 1:
        logger.info(f"Max iterations reached ({max_iterations})")
        return "complete"

    # Check architecture verification result
    arch_verification = state.get("architecture_verification")
    if arch_verification:
        needs_another = arch_verification.get("needs_another_iteration", False)
        coherence = arch_verification.get("coherence_score", 1.0)

        if not needs_another and coherence >= 0.8:
            logger.info(f"Architecture verified (coherence={coherence:.2f}), completing")
            return "complete"
        elif needs_another:
            logger.info(f"Architecture verification requests another iteration (coherence={coherence:.2f})")
            return "continue"

    # Fallback to original logic
    manifest = state.get("edit_manifest")
    if manifest:
        edits_count = len(manifest.get("edits", []))
        if edits_count == 0:
            logger.info("No edits in manifest, completing")
            return "complete"

    logger.info(f"Continuing to iteration {iteration + 2}")
    return "continue"


def increment_iteration(state: Loop3State) -> dict[str, Any]:
    """Increment iteration counter for next loop."""
    return {
        "iteration": state["iteration"] + 1,
    }


def finalize_node(state: Loop3State) -> dict[str, Any]:
    """Mark loop as complete."""
    logger.info("Loop 3 complete")
    return {
        "is_complete": True,
    }


def _format_manifest(manifest: dict) -> str:
    """Format edit manifest for the editor prompt."""
    lines = []

    lines.append(f"Overall Assessment: {manifest.get('overall_assessment', 'N/A')}")
    lines.append("")

    edits = manifest.get("edits", [])
    if edits:
        lines.append("Structural Edits:")
        for i, edit in enumerate(edits, 1):
            edit_type = edit.get("edit_type", "unknown")
            source = edit.get("source_paragraph", "?")
            target = edit.get("target_paragraph")
            notes = edit.get("notes", "")

            lines.append(f"{i}. {edit_type.upper()}")
            lines.append(f"   Source: [P{source}]")
            if target is not None:
                lines.append(f"   Target: [P{target}]")
            lines.append(f"   Notes: {notes}")
            lines.append("")

    todo_markers = manifest.get("todo_markers", [])
    if todo_markers:
        lines.append("TODO Markers to Insert:")
        for i, todo in enumerate(todo_markers, 1):
            lines.append(f"{i}. <!-- TODO: {todo} -->")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Graph Construction
# =============================================================================


def create_loop3_graph():
    """Create Loop 3 StateGraph for structural editing.

    Graph flow:
    - analyze_structure -> validate_edits (if restructuring needed)
    - validate_edits -> apply_edits_programmatic (if valid edits)
                    -> retry_analyze (if missing replacement_text, first time)
                    -> execute_manifest_llm (if all invalid, fallback)
                    -> finalize (if no edits)
    - retry_analyze -> validate_edits (re-validate after retry)
    - apply_edits_programmatic -> verify_application -> verify_architecture -> validate_result
    - execute_manifest_llm -> verify_architecture -> validate_result
    - validate_result -> continue loop or finalize

    Returns a compiled graph ready for execution.
    """
    graph = StateGraph(Loop3State)

    graph.add_node("number_paragraphs", number_paragraphs_node)
    graph.add_node("analyze_structure", analyze_structure_node)
    graph.add_node("validate_edits", validate_edits_node)
    graph.add_node("retry_analyze", retry_analyze_node)
    graph.add_node("apply_edits_programmatic", apply_edits_programmatically_node)
    graph.add_node("verify_application", verify_application_node)
    graph.add_node("verify_architecture", verify_architecture_node)
    graph.add_node("execute_manifest_llm", execute_manifest_node)
    graph.add_node("validate_result", validate_result_node)
    graph.add_node("increment_iteration", increment_iteration)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "number_paragraphs")
    graph.add_edge("number_paragraphs", "analyze_structure")

    graph.add_conditional_edges(
        "analyze_structure",
        route_after_analysis,
        {
            "restructure_needed": "validate_edits",
            "pass_through": "finalize",
        },
    )

    graph.add_conditional_edges(
        "validate_edits",
        route_after_validation,
        {
            "has_valid_edits": "apply_edits_programmatic",
            "needs_retry": "retry_analyze",
            "llm_fallback": "execute_manifest_llm",
            "no_edits": "finalize",
        },
    )

    # Retry goes back to validate_edits
    graph.add_edge("retry_analyze", "validate_edits")

    # Programmatic path: apply -> verify_application -> verify_architecture -> validate_result
    graph.add_edge("apply_edits_programmatic", "verify_application")
    graph.add_edge("verify_application", "verify_architecture")
    graph.add_edge("verify_architecture", "validate_result")

    # LLM fallback also goes through verify_architecture
    graph.add_edge("execute_manifest_llm", "verify_architecture")

    graph.add_conditional_edges(
        "validate_result",
        check_continue,
        {
            "continue": "increment_iteration",
            "complete": "finalize",
        },
    )

    graph.add_edge("increment_iteration", "number_paragraphs")

    graph.add_edge("finalize", END)

    return graph.compile()


# =============================================================================
# Standalone API
# =============================================================================


async def run_loop3_standalone(
    review: str,
    input_data: LitReviewInput,
    max_iterations: int = 3,
    config: dict | None = None,
) -> dict:
    """Run Loop 3 as standalone operation for testing.

    Args:
        review: Current literature review text
        input_data: Original input parameters with topic and research questions
        max_iterations: Maximum number of restructuring iterations
        config: Optional LangGraph config with run_id and run_name for tracing

    Returns:
        Dictionary containing:
            - current_review: Final restructured review
            - is_complete: Whether loop completed successfully
            - iteration: Number of iterations used
            - edit_manifest: Final edit manifest (if any)
    """
    compiled_graph = create_loop3_graph()

    initial_state: Loop3State = {
        "current_review": review,
        "numbered_document": "",
        "paragraph_mapping": {},
        "edit_manifest": None,
        "input": input_data,
        "iteration": 0,
        "max_iterations": max_iterations,
        "is_complete": False,
        "valid_edits": [],
        "invalid_edits": [],
        "needs_retry_edits": [],
        "validation_errors": {},
        "applied_edits": [],
        "fallback_used": False,
        "retry_attempted": False,
        "architecture_verification": None,
        "needs_another_iteration": False,
    }

    logger.info(f"Running Loop 3 standalone with max_iterations={max_iterations}")

    if config:
        final_state = await compiled_graph.ainvoke(initial_state, config=config)
    else:
        final_state = await compiled_graph.ainvoke(initial_state)

    return {
        "current_review": final_state.get("current_review", review),
        "is_complete": final_state.get("is_complete", False),
        "iteration": final_state.get("iteration", 0),
        "edit_manifest": final_state.get("edit_manifest"),
    }
