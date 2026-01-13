"""Loop 5 graph construction and state definition."""

import logging
from dataclasses import dataclass
from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from .result_processing import (
    split_sections_node,
    validate_edits_node,
    apply_edits_node,
    flag_issues_node,
    finalize_node,
)
from .fact_checking import fact_check_node
from .reference_checking import reference_check_node
from ...types import Edit
from ...utils import SectionInfo
from workflows.shared.workflow_state_store import save_workflow_state

logger = logging.getLogger(__name__)


@dataclass
class Loop5Result:
    """Result from Loop 5 fact and reference checking."""

    current_review: str
    changes_summary: str
    human_review_items: list[str]


class Loop5State(TypedDict, total=False):
    """State for Loop 5 fact and reference checking."""

    # Core inputs
    current_review: str
    topic: str

    # Internal state
    sections: list[SectionInfo]
    all_edits: list[Edit]
    valid_edits: list[Edit]
    invalid_edits: list[Edit]
    ambiguous_claims: list[str]
    unaddressed_todos: list[str]
    human_review_items: list[str]
    discarded_todos: list[str]

    # Loop control
    iteration: int
    max_iterations: int
    is_complete: bool


def create_loop5_graph() -> StateGraph:
    """Create Loop 5 StateGraph for fact and reference checking."""
    graph = StateGraph(Loop5State)

    graph.add_node("split_sections", split_sections_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("reference_check", reference_check_node)
    graph.add_node("validate_edits", validate_edits_node)
    graph.add_node("apply_edits", apply_edits_node)
    graph.add_node("flag_issues", flag_issues_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "split_sections")
    graph.add_edge("split_sections", "fact_check")
    graph.add_edge("fact_check", "reference_check")
    graph.add_edge("reference_check", "validate_edits")
    graph.add_edge("validate_edits", "apply_edits")
    graph.add_edge("apply_edits", "flag_issues")
    graph.add_edge("flag_issues", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


@traceable(run_type="chain", name="Loop5_FactReferenceCheck")
async def run_loop5_standalone(
    review: str,
    topic: str,
    quality_settings: dict[str, Any],
    config: dict | None = None,
) -> Loop5Result:
    """Run Loop 5 fact and reference checking.

    Args:
        review: Current document text
        topic: Review topic
        quality_settings: Quality tier settings (max_stages used for iterations)
        config: Optional LangSmith config

    Returns:
        Loop5Result with updated review, summary, and human review items
    """
    graph = create_loop5_graph()

    # Derive max_iterations from quality settings (minimum 1)
    max_iterations = max(1, quality_settings.get("max_stages", 1))

    initial_state: Loop5State = {
        "current_review": review,
        "topic": topic,
        "sections": [],
        "all_edits": [],
        "valid_edits": [],
        "invalid_edits": [],
        "ambiguous_claims": [],
        "unaddressed_todos": [],
        "human_review_items": [],
        "discarded_todos": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "is_complete": False,
    }

    if config:
        result = await graph.ainvoke(initial_state, config=config)
    else:
        result = await graph.ainvoke(initial_state)

    # Build changes summary
    valid_edits = result.get("valid_edits", [])
    invalid_edits = result.get("invalid_edits", [])
    human_items = result.get("human_review_items", [])

    summary_parts = []
    if valid_edits:
        summary_parts.append(f"{len(valid_edits)} edits applied")
    if invalid_edits:
        summary_parts.append(f"{len(invalid_edits)} edits rejected")
    if human_items:
        summary_parts.append(f"{len(human_items)} items flagged for review")

    changes_summary = "; ".join(summary_parts) if summary_parts else "No changes"

    # Save state for analysis
    run_id = config.get("run_id") if config else None
    if run_id:
        save_workflow_state(
            workflow_name="supervision_loop5",
            run_id=str(run_id),
            state={
                "input": {
                    "topic": topic,
                },
                "output": {
                    "changes_summary": changes_summary,
                    "human_review_items": len(human_items),
                },
                "final_state": {
                    "valid_edits": len(valid_edits),
                    "invalid_edits": len(invalid_edits),
                    "ambiguous_claims": len(result.get("ambiguous_claims", [])),
                    "discarded_todos": len(result.get("discarded_todos", [])),
                },
            },
        )

    logger.info(f"Loop 5 complete: {changes_summary}")

    return Loop5Result(
        current_review=result["current_review"],
        changes_summary=changes_summary,
        human_review_items=human_items,
    )
