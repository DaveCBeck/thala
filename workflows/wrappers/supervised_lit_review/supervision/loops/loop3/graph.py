"""Loop 3 graph construction and standalone API.

Two-phase structural editing loop:
- Phase A: Identify structural issues (diagnosis)
- Phase B: Rewrite sections to fix issues (section-rewrite approach)
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from typing_extensions import TypedDict

from workflows.shared.workflow_state_store import save_workflow_state
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


@dataclass
class Loop3Result:
    """Result from running Loop 3 structural editing."""

    current_review: str
    changes_summary: str
    iterations_used: int


class Loop3State(TypedDict, total=False):
    """State for Loop 3 structural editing.

    Uses section-rewrite approach:
    - issue_analysis: Phase A output (StructuralIssueAnalysis)
    - rewrite_manifest: Phase B output (Loop3RewriteManifest)
    """

    # Core inputs
    current_review: str
    topic: str
    iteration: int
    max_iterations: int
    is_complete: bool

    # Document processing (internal)
    numbered_document: str
    paragraph_mapping: dict[int, str]

    # Phase A output (diagnosis)
    issue_analysis: Optional[dict]
    phase_a_complete: bool

    # Phase B output (section rewrites)
    rewrite_manifest: Optional[dict]
    changes_applied: list[str]

    # Verification output
    architecture_verification: Optional[dict]
    needs_another_iteration: bool


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


@traceable(run_type="chain", name="Loop3_StructureCohesion")
async def run_loop3_standalone(
    review: str,
    topic: str,
    quality_settings: dict[str, Any],
    config: dict | None = None,
) -> Loop3Result:
    """Run Loop 3 as standalone operation.

    Args:
        review: Current literature review text
        topic: Research topic for context
        quality_settings: Quality tier settings (max_stages used for iterations)
        config: Optional LangGraph config with run_id and run_name for tracing

    Returns:
        Loop3Result with current_review, changes_summary, iterations_used
    """
    # Derive max_iterations from quality settings (+1 for structural editing)
    # Loop 3 gets an extra iteration because structural issues often
    # require multiple passes to fully resolve
    max_iterations = quality_settings.get("max_stages", 3) + 1

    compiled_graph = create_loop3_graph()
    run_id = config.get("run_id", uuid.uuid4()) if config else uuid.uuid4()

    initial_state: Loop3State = {
        "current_review": review,
        "topic": topic,
        "numbered_document": "",
        "paragraph_mapping": {},
        "iteration": 0,
        "max_iterations": max_iterations,
        "is_complete": False,
        "issue_analysis": None,
        "phase_a_complete": False,
        "rewrite_manifest": None,
        "changes_applied": [],
        "architecture_verification": None,
        "needs_another_iteration": False,
    }

    logger.info(f"Loop 3 starting with max_iterations={max_iterations}")

    if config:
        final_state = await compiled_graph.ainvoke(initial_state, config=config)
    else:
        final_state = await compiled_graph.ainvoke(initial_state)

    # Build changes summary
    iterations_used = final_state.get("iteration", 0)
    rewrite_manifest = final_state.get("rewrite_manifest", {})
    changes_applied = final_state.get("changes_applied", [])

    if changes_applied:
        changes_summary = f"Resolved {len(changes_applied)} structural issues"
    elif rewrite_manifest and rewrite_manifest.get("rewrites"):
        changes_summary = (
            f"Applied {len(rewrite_manifest['rewrites'])} section rewrites"
        )
    else:
        changes_summary = "No structural changes needed"

    # Save state for analysis (dev mode only)
    save_workflow_state(
        workflow_name="supervision_loop3",
        run_id=str(run_id),
        state={
            "input": {
                "topic": topic,
                "review_length": len(review),
                "max_iterations": max_iterations,
            },
            "output": {
                "changes_summary": changes_summary,
                "iterations_used": iterations_used,
            },
            "final_state": {
                "rewrite_manifest": rewrite_manifest,
                "changes_applied": changes_applied,
                "architecture_verification": final_state.get(
                    "architecture_verification"
                ),
            },
        },
    )

    return Loop3Result(
        current_review=final_state.get("current_review", review),
        changes_summary=changes_summary,
        iterations_used=iterations_used,
    )
