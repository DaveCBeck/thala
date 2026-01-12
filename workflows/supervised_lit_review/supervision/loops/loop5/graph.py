"""Loop 5 graph construction and state definition."""

from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

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


class Loop5State(TypedDict):
    """State for Loop 5 fact and reference checking."""

    current_review: str
    paper_summaries: dict[str, Any]
    zotero_keys: dict[str, str]
    sections: list[SectionInfo]
    all_edits: list[Edit]
    valid_edits: list[Edit]
    invalid_edits: list[Edit]
    ambiguous_claims: list[str]
    unaddressed_todos: list[str]
    human_review_items: list[str]
    discarded_todos: list[str]
    iteration: int
    max_iterations: int
    is_complete: bool
    topic: str
    verify_todos_enabled: bool
    verify_zotero: bool
    verified_citation_keys: set[str]


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


async def run_loop5_standalone(
    review: str,
    paper_summaries: dict,
    zotero_keys: dict,
    max_iterations: int = 1,
    config: dict | None = None,
    topic: str = "",
    verify_todos_enabled: bool = True,
    verify_zotero: bool = True,
) -> dict:
    """Run Loop 5 as standalone operation for testing."""
    graph = create_loop5_graph()

    initial_state = Loop5State(
        current_review=review,
        paper_summaries=paper_summaries,
        zotero_keys=zotero_keys,
        sections=[],
        all_edits=[],
        valid_edits=[],
        invalid_edits=[],
        ambiguous_claims=[],
        unaddressed_todos=[],
        human_review_items=[],
        discarded_todos=[],
        iteration=0,
        max_iterations=max_iterations,
        is_complete=False,
        topic=topic,
        verify_todos_enabled=verify_todos_enabled,
        verify_zotero=verify_zotero,
        verified_citation_keys=set(),
    )

    if config:
        result = await graph.ainvoke(initial_state, config=config)
    else:
        result = await graph.ainvoke(initial_state)

    return {
        "current_review": result["current_review"],
        "human_review_items": result.get("human_review_items", []),
        "discarded_todos": result.get("discarded_todos", []),
        "ambiguous_claims": result.get("ambiguous_claims", []),
        "unaddressed_todos": result.get("unaddressed_todos", []),
        "valid_edits": result.get("valid_edits", []),
        "invalid_edits": result.get("invalid_edits", []),
        "verified_citation_keys": result.get("verified_citation_keys", set()),
    }
