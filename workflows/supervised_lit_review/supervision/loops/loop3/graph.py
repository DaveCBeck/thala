"""Loop 3 graph construction and standalone API.

Two-phase structural editing loop:
- Phase A: Identify structural issues (diagnosis)
- Phase B: Rewrite sections to fix issues (new approach - replaces edit generation)
"""

import logging
from typing_extensions import TypedDict
from typing import Optional

from langgraph.graph import StateGraph, START, END

from workflows.academic_lit_review.state import LitReviewInput
from .analyzer import analyze_structure_phase_a_node
from .section_rewriter import rewrite_sections_for_issues_node
from .verification import verify_architecture_node
from .routing import route_after_phase_a, check_continue_rewrite
from .utils import (
    number_paragraphs_node,
    validate_result_node,
    increment_iteration,
    finalize_node,
)

logger = logging.getLogger(__name__)


class Loop3State(TypedDict):
    """State for Loop 3 structural editing.

    Uses section-rewrite approach (replaces edit-based approach):
    - issue_analysis: Phase A output (StructuralIssueAnalysis)
    - rewrite_manifest: Phase B output (Loop3RewriteManifest)
    """

    current_review: str
    numbered_document: str
    paragraph_mapping: dict[int, str]
    input: LitReviewInput
    iteration: int
    max_iterations: int
    is_complete: bool

    issue_analysis: Optional[dict]
    phase_a_complete: bool

    rewrite_manifest: Optional[dict]
    changes_applied: list[str]

    architecture_verification: Optional[dict]
    needs_another_iteration: bool

    zotero_keys: dict[str, str]
    zotero_key_sources: dict[str, dict]

    edit_manifest: Optional[dict]
    applied_edits: list[str]


def create_loop3_graph():
    """Create Loop 3 StateGraph for section-rewrite based structural editing.

    NEW section-rewrite graph flow:
    - number_paragraphs -> phase_a_identify_issues (diagnosis)
    - phase_a_identify_issues -> phase_b_rewrite_sections (if issues found)
                              -> finalize (if pass-through)
    - phase_b_rewrite_sections -> verify_architecture -> validate_result
    - validate_result -> continue loop or finalize

    The section-rewrite approach:
    1. Phase A: Identify structural issues (same as before)
    2. Phase B: Rewrite affected sections directly (replaces edit specification)

    This eliminates the "identifies issues but can't generate valid edits" failure mode.

    Returns a compiled graph ready for execution.
    """
    graph = StateGraph(Loop3State)

    graph.add_node("number_paragraphs", number_paragraphs_node)
    graph.add_node("phase_a_identify_issues", analyze_structure_phase_a_node)
    graph.add_node("phase_b_rewrite_sections", rewrite_sections_for_issues_node)
    graph.add_node("verify_architecture", verify_architecture_node)
    graph.add_node("validate_result", validate_result_node)
    graph.add_node("increment_iteration", increment_iteration)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "number_paragraphs")
    graph.add_edge("number_paragraphs", "phase_a_identify_issues")

    graph.add_conditional_edges(
        "phase_a_identify_issues",
        route_after_phase_a,
        {
            "rewrite_sections": "phase_b_rewrite_sections",
            "pass_through": "finalize",
        },
    )

    graph.add_edge("phase_b_rewrite_sections", "verify_architecture")
    graph.add_edge("verify_architecture", "validate_result")

    graph.add_conditional_edges(
        "validate_result",
        check_continue_rewrite,
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
    zotero_keys: dict[str, str] | None = None,
    zotero_key_sources: dict[str, dict] | None = None,
) -> dict:
    """Run Loop 3 as standalone operation for testing.

    Args:
        review: Current literature review text
        input_data: Original input parameters with topic and research questions
        max_iterations: Maximum number of restructuring iterations
        config: Optional LangGraph config with run_id and run_name for tracing
        zotero_keys: DOI -> citation key mapping for citation validation
        zotero_key_sources: Citation key -> metadata for citation validation

    Returns:
        Dictionary containing:
            - current_review: Final restructured review
            - is_complete: Whether loop completed successfully
            - iteration: Number of iterations used
            - rewrite_manifest: Final rewrite manifest (if any)
    """
    compiled_graph = create_loop3_graph()

    initial_state: Loop3State = {
        "current_review": review,
        "numbered_document": "",
        "paragraph_mapping": {},
        "input": input_data,
        "iteration": 0,
        "max_iterations": max_iterations,
        "is_complete": False,
        "issue_analysis": None,
        "phase_a_complete": False,
        "rewrite_manifest": None,
        "changes_applied": [],
        "architecture_verification": None,
        "needs_another_iteration": False,
        "zotero_keys": zotero_keys or {},
        "zotero_key_sources": zotero_key_sources or {},
        "edit_manifest": None,
        "applied_edits": [],
    }

    logger.info(f"Loop 3 starting with max_iterations={max_iterations}")

    if config:
        final_state = await compiled_graph.ainvoke(initial_state, config=config)
    else:
        final_state = await compiled_graph.ainvoke(initial_state)

    return {
        "current_review": final_state.get("current_review", review),
        "is_complete": final_state.get("is_complete", False),
        "iteration": final_state.get("iteration", 0),
        "rewrite_manifest": final_state.get("rewrite_manifest"),
        "edit_manifest": final_state.get("edit_manifest"),
    }
