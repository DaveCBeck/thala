"""Loop 5: Fact and reference checking."""

import logging
from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from workflows.shared.llm_utils import get_llm, ModelTier

logger = logging.getLogger(__name__)
from ..types import Edit, DocumentEdits
from ..prompts import (
    LOOP5_FACT_CHECK_SYSTEM,
    LOOP5_FACT_CHECK_USER,
    LOOP5_REF_CHECK_SYSTEM,
    LOOP5_REF_CHECK_USER,
)
from ..utils import (
    split_into_sections,
    validate_edits,
    apply_edits,
    SectionInfo,
    EditValidationResult,
)


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
    iteration: int
    max_iterations: int
    is_complete: bool


def split_sections_node(state: Loop5State) -> dict[str, Any]:
    """Split document into sections for sequential checking."""
    sections = split_into_sections(state["current_review"])
    logger.info(f"Loop 5: Split document into {len(sections)} sections for checking")
    return {"sections": sections}


async def fact_check_node(state: Loop5State) -> dict[str, Any]:
    """Sequential fact checking across all sections using Haiku."""
    logger.info(f"Loop 5: Starting fact checking across {len(state['sections'])} sections")
    llm = get_llm(tier=ModelTier.HAIKU, max_tokens=4096)
    structured_llm = llm.with_structured_output(DocumentEdits)

    all_edits: list[Edit] = []
    all_ambiguous: list[str] = []

    # Format paper summaries for context
    paper_summaries_text = _format_paper_summaries(state["paper_summaries"])

    for section in state["sections"]:
        messages = [
            {"role": "system", "content": LOOP5_FACT_CHECK_SYSTEM},
            {
                "role": "user",
                "content": LOOP5_FACT_CHECK_USER.format(
                    section_content=section["section_content"],
                    full_document=state["current_review"],
                    paper_summaries=paper_summaries_text,
                ),
            },
        ]

        result: DocumentEdits = await structured_llm.ainvoke(messages)
        all_edits.extend(result.edits)
        all_ambiguous.extend(result.ambiguous_claims)

    logger.info(f"Loop 5: Fact check found {len(all_edits)} edits, {len(all_ambiguous)} ambiguous claims")
    return {
        "all_edits": all_edits,
        "ambiguous_claims": all_ambiguous,
    }


async def reference_check_node(state: Loop5State) -> dict[str, Any]:
    """Sequential reference checking across all sections using Haiku."""
    logger.info(f"Loop 5: Starting reference checking across {len(state['sections'])} sections")
    llm = get_llm(tier=ModelTier.HAIKU, max_tokens=4096)
    structured_llm = llm.with_structured_output(DocumentEdits)

    all_edits: list[Edit] = state.get("all_edits", []).copy()
    all_todos: list[str] = []

    # Format citation keys for checking
    citation_keys_text = _format_citation_keys(state["zotero_keys"])
    paper_summaries_text = _format_paper_summaries(state["paper_summaries"])

    for section in state["sections"]:
        messages = [
            {"role": "system", "content": LOOP5_REF_CHECK_SYSTEM},
            {
                "role": "user",
                "content": LOOP5_REF_CHECK_USER.format(
                    section_content=section["section_content"],
                    full_document=state["current_review"],
                    citation_keys=citation_keys_text,
                    paper_summaries=paper_summaries_text,
                ),
            },
        ]

        result: DocumentEdits = await structured_llm.ainvoke(messages)
        all_edits.extend(result.edits)
        all_todos.extend(result.unaddressed_todos)

    logger.info(f"Loop 5: Reference check complete, total edits: {len(all_edits)}, unaddressed TODOs: {len(all_todos)}")
    return {
        "all_edits": all_edits,
        "unaddressed_todos": all_todos,
    }


def validate_edits_node(state: Loop5State) -> dict[str, Any]:
    """Validate that all edit find strings exist and are unambiguous."""
    validation_result: EditValidationResult = validate_edits(
        state["current_review"], state["all_edits"]
    )

    # Format invalid edits as human review items
    human_items = []
    for idx, edit in enumerate(validation_result["invalid_edits"]):
        error_type = validation_result["errors"].get(str(idx), "unknown")
        human_items.append(
            f"Invalid edit ({error_type}): '{edit.find[:50]}...' -> '{edit.replace[:50]}...'"
        )

    logger.info(
        f"Loop 5: Validated edits - {len(validation_result['valid_edits'])} valid, "
        f"{len(validation_result['invalid_edits'])} invalid"
    )

    return {
        "valid_edits": validation_result["valid_edits"],
        "invalid_edits": validation_result["invalid_edits"],
        "human_review_items": human_items,
    }


def apply_edits_node(state: Loop5State) -> dict[str, Any]:
    """Apply validated edits to the document."""
    # Filter edits: only fact_correction, citation_fix, clarity (ignore copy-editing)
    allowed_types = {"fact_correction", "citation_fix", "clarity"}
    filtered_edits = [
        edit for edit in state["valid_edits"] if edit.edit_type in allowed_types
    ]

    updated_review = apply_edits(state["current_review"], filtered_edits)

    logger.info(f"Loop 5: Applied {len(filtered_edits)} edits to document")

    return {"current_review": updated_review}


def flag_issues_node(state: Loop5State) -> dict[str, Any]:
    """Collect ambiguous claims and unaddressed TODOs for human review."""
    human_items = state.get("human_review_items", []).copy()

    # Add ambiguous claims
    for claim in state.get("ambiguous_claims", []):
        human_items.append(f"Ambiguous claim: {claim}")

    # Add unaddressed TODOs
    for todo in state.get("unaddressed_todos", []):
        human_items.append(f"Unaddressed TODO: {todo}")

    return {"human_review_items": human_items}


def finalize_node(state: Loop5State) -> dict[str, Any]:
    """Mark loop as complete."""
    logger.info("Loop 5: Fact and reference checking complete")
    return {"is_complete": True}


def _format_paper_summaries(paper_summaries: dict[str, Any]) -> str:
    """Format paper summaries for prompt context."""
    if not paper_summaries:
        return "No paper summaries available."

    lines = []
    for doi, summary in paper_summaries.items():
        lines.append(f"DOI: {doi}")
        lines.append(f"Title: {summary.get('title', 'N/A')}")
        lines.append(f"Authors: {', '.join(summary.get('authors', []))}")
        lines.append(f"Year: {summary.get('year', 'N/A')}")
        lines.append(f"Summary: {summary.get('short_summary', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def _format_citation_keys(zotero_keys: dict[str, str]) -> str:
    """Format citation keys for reference checking."""
    if not zotero_keys:
        return "No citation keys available."

    lines = []
    for doi, key in zotero_keys.items():
        lines.append(f"[@{key}] -> {doi}")

    return "\n".join(lines)


def create_loop5_graph() -> StateGraph:
    """Create Loop 5 StateGraph for fact and reference checking.

    Graph flow:
        START → split_sections → fact_check → reference_check
            → validate_edits → apply_edits → flag_issues → finalize → END
    """
    graph = StateGraph(Loop5State)

    # Add nodes
    graph.add_node("split_sections", split_sections_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("reference_check", reference_check_node)
    graph.add_node("validate_edits", validate_edits_node)
    graph.add_node("apply_edits", apply_edits_node)
    graph.add_node("flag_issues", flag_issues_node)
    graph.add_node("finalize", finalize_node)

    # Add edges
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
) -> dict:
    """Run Loop 5 as standalone operation for testing.

    Args:
        review: Literature review text to check
        paper_summaries: Paper summaries for fact checking
        zotero_keys: DOI -> Zotero key mapping for citation checking
        max_iterations: Maximum iterations (usually 1 for fact checking)

    Returns:
        Dict with:
            - current_review: Updated review text
            - human_review_items: Issues flagged for human review
            - ambiguous_claims: Claims that need verification
            - unaddressed_todos: TODOs that couldn't be resolved
            - valid_edits: Successfully applied edits
            - invalid_edits: Edits that couldn't be applied
    """
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
        iteration=0,
        max_iterations=max_iterations,
        is_complete=False,
    )

    result = await graph.ainvoke(initial_state)

    return {
        "current_review": result["current_review"],
        "human_review_items": result.get("human_review_items", []),
        "ambiguous_claims": result.get("ambiguous_claims", []),
        "unaddressed_todos": result.get("unaddressed_todos", []),
        "valid_edits": result.get("valid_edits", []),
        "invalid_edits": result.get("invalid_edits", []),
    }
