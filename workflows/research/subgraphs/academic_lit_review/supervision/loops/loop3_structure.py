"""Loop 3: Structure and Cohesion - Structural editing with two-agent pattern."""

import logging
from typing import Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from workflows.shared.llm_utils import ModelTier, get_llm
from workflows.research.subgraphs.academic_lit_review.state import LitReviewInput
from workflows.research.subgraphs.academic_lit_review.supervision.types import (
    EditManifest,
    StructuralEdit,
)
from workflows.research.subgraphs.academic_lit_review.supervision.prompts import (
    LOOP3_ANALYST_SYSTEM,
    LOOP3_ANALYST_USER,
    LOOP3_EDITOR_SYSTEM,
    LOOP3_EDITOR_USER,
)
from workflows.research.subgraphs.academic_lit_review.supervision.utils import (
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

    # Validation tracking (new fields)
    valid_edits: list[dict]
    invalid_edits: list[dict]
    validation_errors: dict[int, str]
    applied_edits: list[str]
    fallback_used: bool


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
    Includes retry logic for parsing failures.
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

    llm = get_llm(
        tier=ModelTier.OPUS,
        thinking_budget=8000,
        max_tokens=12096,
    )

    MAX_RETRIES = 2
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            structured_llm = llm.with_structured_output(EditManifest, method="json_schema")
            messages = [
                {"role": "system", "content": LOOP3_ANALYST_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]

            manifest: EditManifest = await structured_llm.ainvoke(messages)

            # Warn if needs_restructuring but no edits
            if manifest.needs_restructuring and len(manifest.edits) == 0 and len(manifest.todo_markers) == 0:
                logger.warning(
                    "Manifest indicates needs_restructuring=True but edits and todo_markers are empty. "
                    f"Assessment: {manifest.overall_assessment[:100]}..."
                )

            logger.info(
                f"Loop 3 Analyst (attempt {attempt + 1}): needs_restructuring={manifest.needs_restructuring}, "
                f"edits={len(manifest.edits)}, todos={len(manifest.todo_markers)}"
            )

            return {
                "edit_manifest": manifest.model_dump(),
            }

        except Exception as e:
            last_error = e
            logger.warning(f"Structure analysis attempt {attempt + 1} failed: {e}")
            continue

    # All retries failed
    logger.error(f"Structure analysis failed after {MAX_RETRIES} attempts: {last_error}")
    return {
        "edit_manifest": {
            "edits": [],
            "todo_markers": [],
            "overall_assessment": f"Analysis failed after {MAX_RETRIES} attempts: {last_error}",
            "needs_restructuring": False,
        },
    }


def route_after_analysis(state: Loop3State) -> str:
    """Route based on whether restructuring is needed."""
    manifest = state.get("edit_manifest")

    if not manifest:
        return "pass_through"

    needs_restructuring = manifest.get("needs_restructuring", False)

    if needs_restructuring:
        logger.info("Routing to validate_edits (restructuring needed)")
        return "restructure_needed"
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
            "validation_errors": {},
        }

    # Parse edits to StructuralEdit objects
    edits = [StructuralEdit(**e) for e in manifest.get("edits", [])]

    result = validate_structural_edits(paragraph_mapping, edits)

    logger.info(
        f"Edit validation: {len(result['valid_edits'])} valid, "
        f"{len(result['invalid_edits'])} invalid"
    )

    if result["invalid_edits"]:
        for idx, error in result["errors"].items():
            logger.warning(f"Invalid edit {idx}: {error}")

    return {
        "valid_edits": [e.model_dump() for e in result["valid_edits"]],
        "invalid_edits": [e.model_dump() for e in result["invalid_edits"]],
        "validation_errors": result["errors"],
    }


def route_after_validation(state: Loop3State) -> str:
    """Route based on validation results."""
    valid_edits = state.get("valid_edits", [])
    invalid_edits = state.get("invalid_edits", [])
    manifest = state.get("edit_manifest")

    if not manifest or not manifest.get("needs_restructuring"):
        return "no_edits"

    if valid_edits:
        logger.info(f"Routing to programmatic application ({len(valid_edits)} valid edits)")
        return "has_valid_edits"

    if invalid_edits:
        # All edits failed validation, try LLM fallback
        logger.warning(
            f"All {len(invalid_edits)} edits invalid, falling back to LLM execution"
        )
        return "llm_fallback"

    # No edits at all
    return "no_edits"


def apply_edits_programmatically_node(state: Loop3State) -> dict[str, Any]:
    """Apply validated edits programmatically using paragraph mapping."""
    paragraph_mapping = state.get("paragraph_mapping", {})
    valid_edits = state.get("valid_edits", [])

    if not valid_edits:
        logger.info("No valid edits to apply programmatically")
        return {"fallback_used": False}

    # Convert back to StructuralEdit objects
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


async def execute_manifest_node(state: Loop3State) -> dict[str, Any]:
    """Execute the edit manifest to restructure the document (LLM fallback).

    Used when programmatic edit application fails or isn't possible.
    Uses Opus to carefully execute structural changes while preserving
    citations and academic formatting.
    """
    numbered_doc = state["numbered_document"]
    manifest = state.get("edit_manifest")

    if not manifest:
        logger.warning("No manifest to execute")
        return {"fallback_used": True}

    # Format manifest for prompt
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

        # Extract text from response
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
        # Return unchanged on error
        return {"fallback_used": True}


def validate_result_node(state: Loop3State) -> dict[str, Any]:
    """Strip paragraph numbers and validate the restructured output."""
    current_review = state["current_review"]

    # Remove paragraph numbers if present
    cleaned_review = strip_paragraph_numbers(current_review)

    # Basic validation
    if not cleaned_review or len(cleaned_review) < 100:
        logger.warning("Validation failed: output too short or empty")
        # Revert to original if validation fails
        original_numbered = state.get("numbered_document", "")
        if original_numbered:
            cleaned_review = strip_paragraph_numbers(original_numbered)

    logger.info(f"Validated result: {len(cleaned_review)} characters")

    return {
        "current_review": cleaned_review,
    }


def check_continue(state: Loop3State) -> str:
    """Check if we should continue or complete the loop."""
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]

    if iteration >= max_iterations - 1:
        logger.info(f"Max iterations reached ({max_iterations})")
        return "complete"

    # Check if previous iteration made changes
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
                    -> execute_manifest_llm (if all invalid, fallback)
                    -> finalize (if no edits)
    - apply_edits_programmatic -> verify_application -> validate_result
    - execute_manifest_llm -> validate_result
    - validate_result -> continue loop or finalize

    Returns a compiled graph ready for execution.
    """
    graph = StateGraph(Loop3State)

    # Add nodes
    graph.add_node("number_paragraphs", number_paragraphs_node)
    graph.add_node("analyze_structure", analyze_structure_node)
    graph.add_node("validate_edits", validate_edits_node)  # NEW
    graph.add_node("apply_edits_programmatic", apply_edits_programmatically_node)  # NEW
    graph.add_node("verify_application", verify_application_node)  # NEW
    graph.add_node("execute_manifest_llm", execute_manifest_node)  # RENAMED (LLM fallback)
    graph.add_node("validate_result", validate_result_node)
    graph.add_node("increment_iteration", increment_iteration)
    graph.add_node("finalize", finalize_node)

    # Build flow
    graph.add_edge(START, "number_paragraphs")
    graph.add_edge("number_paragraphs", "analyze_structure")

    # Route after analysis
    graph.add_conditional_edges(
        "analyze_structure",
        route_after_analysis,
        {
            "restructure_needed": "validate_edits",  # Go to validation first
            "pass_through": "finalize",
        },
    )

    # Route after validation
    graph.add_conditional_edges(
        "validate_edits",
        route_after_validation,
        {
            "has_valid_edits": "apply_edits_programmatic",
            "llm_fallback": "execute_manifest_llm",  # Use LLM if validation failed
            "no_edits": "finalize",
        },
    )

    # Programmatic path
    graph.add_edge("apply_edits_programmatic", "verify_application")
    graph.add_edge("verify_application", "validate_result")

    # LLM fallback path
    graph.add_edge("execute_manifest_llm", "validate_result")

    # Check if should continue
    graph.add_conditional_edges(
        "validate_result",
        check_continue,
        {
            "continue": "increment_iteration",
            "complete": "finalize",
        },
    )

    # Loop back to number paragraphs
    graph.add_edge("increment_iteration", "number_paragraphs")

    # End
    graph.add_edge("finalize", END)

    return graph.compile()


# =============================================================================
# Standalone API
# =============================================================================


async def run_loop3_standalone(
    review: str,
    input_data: LitReviewInput,
    max_iterations: int = 3,
) -> dict:
    """Run Loop 3 as standalone operation for testing.

    Args:
        review: Current literature review text
        input_data: Original input parameters with topic and research questions
        max_iterations: Maximum number of restructuring iterations

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
        # New validation tracking fields
        "valid_edits": [],
        "invalid_edits": [],
        "validation_errors": {},
        "applied_edits": [],
        "fallback_used": False,
    }

    logger.info(f"Running Loop 3 standalone with max_iterations={max_iterations}")

    final_state = await compiled_graph.ainvoke(initial_state)

    return {
        "current_review": final_state.get("current_review", review),
        "is_complete": final_state.get("is_complete", False),
        "iteration": final_state.get("iteration", 0),
        "edit_manifest": final_state.get("edit_manifest"),
    }
