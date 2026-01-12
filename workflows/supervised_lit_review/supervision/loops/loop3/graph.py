"""Loop 3 graph construction and standalone API.

Two-phase structural editing loop:
- Phase A: Identify structural issues (diagnosis)
- Phase B: Generate concrete edits based on identified issues (prescription)
"""

import logging
from typing_extensions import TypedDict
from typing import Optional

from langgraph.graph import StateGraph, START, END

from workflows.academic_lit_review.state import LitReviewInput
from .analyzer import analyze_structure_phase_a_node, generate_edits_phase_b_node
from .fallback import retry_analyze_node, execute_manifest_node
from .validator import validate_edits_node, apply_edits_programmatically_node, verify_application_node
from .verification import verify_architecture_node
from .routing import route_after_phase_a, route_after_analysis, route_after_validation, check_continue
from .utils import (
    number_paragraphs_node,
    validate_result_node,
    increment_iteration,
    finalize_node,
)

logger = logging.getLogger(__name__)


class Loop3State(TypedDict):
    """State for Loop 3 structural editing.

    Two-phase analysis fields:
    - issue_analysis: Phase A output (StructuralIssueAnalysis)
    - phase_a_complete: Whether Phase A has run
    """

    current_review: str
    numbered_document: str
    paragraph_mapping: dict[int, str]
    edit_manifest: Optional[dict]
    input: LitReviewInput
    iteration: int
    max_iterations: int
    is_complete: bool

    issue_analysis: Optional[dict]
    phase_a_complete: bool

    valid_edits: list[dict]
    invalid_edits: list[dict]
    needs_retry_edits: list[dict]
    validation_errors: dict[int, str]
    applied_edits: list[str]
    fallback_used: bool
    retry_attempted: bool

    architecture_verification: Optional[dict]
    needs_another_iteration: bool


def create_loop3_graph():
    """Create Loop 3 StateGraph for two-phase structural editing.

    NEW two-phase graph flow:
    - number_paragraphs -> phase_a_identify_issues (diagnosis)
    - phase_a_identify_issues -> phase_b_generate_edits (if issues found)
                              -> finalize (if pass-through)
    - phase_b_generate_edits -> validate_edits
    - validate_edits -> apply_edits_programmatic (if valid edits)
                    -> retry_analyze (if missing replacement_text, first time)
                    -> execute_manifest_llm (if all invalid, fallback)
                    -> finalize (if no edits)
    - retry_analyze -> validate_edits (re-validate after retry)
    - apply_edits_programmatic -> verify_application -> verify_architecture -> validate_result
    - execute_manifest_llm -> verify_architecture -> validate_result
    - validate_result -> continue loop or finalize

    The two-phase approach separates:
    1. Phase A: Issue identification (diagnosis)
    2. Phase B: Edit generation (prescription)

    This eliminates the "identifies issues but returns empty edits" failure mode.

    Returns a compiled graph ready for execution.
    """
    graph = StateGraph(Loop3State)

    graph.add_node("number_paragraphs", number_paragraphs_node)
    graph.add_node("phase_a_identify_issues", analyze_structure_phase_a_node)
    graph.add_node("phase_b_generate_edits", generate_edits_phase_b_node)
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
    graph.add_edge("number_paragraphs", "phase_a_identify_issues")

    graph.add_conditional_edges(
        "phase_a_identify_issues",
        route_after_phase_a,
        {
            "generate_edits": "phase_b_generate_edits",
            "pass_through": "finalize",
        },
    )

    graph.add_edge("phase_b_generate_edits", "validate_edits")

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

    graph.add_edge("retry_analyze", "validate_edits")

    graph.add_edge("apply_edits_programmatic", "verify_application")
    graph.add_edge("verify_application", "verify_architecture")
    graph.add_edge("verify_architecture", "validate_result")

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
        "issue_analysis": None,
        "phase_a_complete": False,
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
