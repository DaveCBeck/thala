"""Loop 4: Section-Level Deep Editing with parallel processing."""

import asyncio
import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from workflows.research.subgraphs.academic_lit_review.state import (
    LitReviewInput,
    PaperSummary,
)
from workflows.research.subgraphs.academic_lit_review.supervision.types import (
    SectionEditResult,
    HolisticReviewResult,
)
from workflows.research.subgraphs.academic_lit_review.supervision.prompts import (
    LOOP4_SECTION_EDITOR_SYSTEM,
    LOOP4_SECTION_EDITOR_USER,
    LOOP4_HOLISTIC_SYSTEM,
    LOOP4_HOLISTIC_USER,
)
from workflows.research.subgraphs.academic_lit_review.supervision.utils import (
    split_into_sections,
    SectionInfo,
)
from workflows.shared.llm_utils import get_llm, ModelTier

logger = logging.getLogger(__name__)


class Loop4State(TypedDict):
    """State schema for Loop 4 section-level editing."""

    # Input content
    current_review: str
    paper_summaries: dict[str, PaperSummary]
    input: LitReviewInput

    # Section tracking
    sections: list[SectionInfo]
    section_results: dict[str, SectionEditResult]
    editor_notes: list[str]

    # Holistic review
    holistic_result: Optional[HolisticReviewResult]
    flagged_sections: list[str]

    # Iteration control
    iteration: int
    max_iterations: int
    is_complete: bool


def split_sections_node(state: dict[str, Any]) -> dict[str, Any]:
    """Split document into sections for parallel editing.

    On first pass: split all sections.
    On subsequent passes: only split flagged sections.
    """
    current_review = state.get("current_review", "")
    iteration = state.get("iteration", 0)
    flagged = state.get("flagged_sections", [])

    if iteration == 0:
        # First pass: split all sections
        sections = split_into_sections(current_review, max_tokens=5000)
        logger.info(f"Split document into {len(sections)} sections for initial editing")
    else:
        # Subsequent passes: only sections flagged by holistic review
        all_sections = split_into_sections(current_review, max_tokens=5000)
        sections = [s for s in all_sections if s["section_id"] in flagged]
        logger.info(f"Re-editing {len(sections)} flagged sections")

    return {"sections": sections}


async def parallel_edit_sections_node(state: dict[str, Any]) -> dict[str, Any]:
    """Phase A: Parallel section editing with concurrent Opus calls."""
    sections = state.get("sections", [])
    full_document = state.get("current_review", "")
    paper_summaries = state.get("paper_summaries", {})

    # Limit to 5 concurrent editors
    semaphore = asyncio.Semaphore(5)

    async def edit_section(section: SectionInfo) -> tuple[str, SectionEditResult]:
        """Edit a single section with Opus."""
        async with semaphore:
            section_id = section["section_id"]
            section_content = section["section_content"]

            # Extract TODOs in this section
            todos = [
                line.strip()
                for line in section_content.split("\n")
                if "<!-- TODO:" in line
            ]

            # Format paper summaries for prompt
            summary_text = "\n\n".join([
                f"[@{doi}] {s['title']} ({s['year']})\n{s['short_summary']}"
                for doi, s in paper_summaries.items()
            ])

            # Call Opus for section editing
            llm = get_llm(ModelTier.OPUS)

            user_prompt = LOOP4_SECTION_EDITOR_USER.format(
                full_document=full_document,
                section_id=section_id,
                section_content=section_content,
                paper_summaries=summary_text,
                todos_in_section="\n".join(todos) if todos else "None"
            )

            messages = [
                {"role": "system", "content": LOOP4_SECTION_EDITOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]

            response = await llm.with_structured_output(SectionEditResult).ainvoke(
                messages
            )

            logger.info(f"Edited section '{section_id}' (confidence: {response.confidence:.2f})")
            return section_id, response

    # Execute all section edits in parallel
    edit_tasks = [edit_section(s) for s in sections]
    results = await asyncio.gather(*edit_tasks)

    # Convert to dict
    section_results = {section_id: result for section_id, result in results}

    # Collect editor notes
    editor_notes = [
        f"[{section_id}] {result.notes}"
        for section_id, result in section_results.items()
        if result.notes
    ]

    logger.info(f"Completed parallel editing of {len(section_results)} sections")

    return {
        "section_results": section_results,
        "editor_notes": editor_notes,
    }


def reassemble_document_node(state: dict[str, Any]) -> dict[str, Any]:
    """Reassemble document from edited sections."""
    current_review = state.get("current_review", "")
    sections = state.get("sections", [])
    section_results = state.get("section_results", {})
    iteration = state.get("iteration", 0)

    if iteration == 0:
        # First pass: replace all sections with edited versions
        edited_content = []
        for section in sections:
            section_id = section["section_id"]
            if section_id in section_results:
                edited_content.append(section_results[section_id].edited_content)
            else:
                # Shouldn't happen, but fallback to original
                edited_content.append(section["section_content"])

        updated_review = "\n\n".join(edited_content)
    else:
        # Subsequent passes: replace only flagged sections
        # Sort sections by start_line in REVERSE order to preserve indices
        # When we modify from the end first, earlier indices remain valid
        lines = current_review.split("\n")

        sorted_sections = sorted(
            [s for s in sections if s["section_id"] in section_results],
            key=lambda s: s["start_line"],
            reverse=True  # Process from end to beginning
        )

        for section in sorted_sections:
            section_id = section["section_id"]
            start_line = section["start_line"]
            end_line = section["end_line"]
            edited_lines = section_results[section_id].edited_content.split("\n")

            # Splice in edited content (safe because we process from end first)
            lines = lines[:start_line] + edited_lines + lines[end_line + 1:]

        updated_review = "\n".join(lines)

    logger.info(f"Reassembled document: {len(updated_review)} chars")

    return {"current_review": updated_review}


async def holistic_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """Phase B: Holistic coherence review with single Opus call."""
    document = state.get("current_review", "")
    editor_notes = state.get("editor_notes", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    llm = get_llm(ModelTier.OPUS)

    user_prompt = LOOP4_HOLISTIC_USER.format(
        document=document,
        editor_notes="\n".join(editor_notes),
        iteration=iteration + 1,
        max_iterations=max_iterations
    )

    messages = [
        {"role": "system", "content": LOOP4_HOLISTIC_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    result = await llm.with_structured_output(HolisticReviewResult).ainvoke(
        messages
    )

    logger.info(
        f"Holistic review: {len(result.sections_approved)} approved, "
        f"{len(result.sections_flagged)} flagged (coherence: {result.overall_coherence_score:.2f})"
    )

    return {
        "holistic_result": result,
        "flagged_sections": result.sections_flagged,
        "iteration": iteration + 1,
    }


def route_after_holistic(state: dict[str, Any]) -> str:
    """Route based on holistic review results."""
    holistic = state.get("holistic_result")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if not holistic:
        return "finalize"

    # Check if we have flagged sections and haven't hit max iterations
    has_flagged = len(holistic.sections_flagged) > 0
    can_continue = iteration < max_iterations

    if has_flagged and can_continue:
        logger.info(f"Continuing to re-edit {len(holistic.sections_flagged)} flagged sections")
        return "split_sections"
    else:
        if not has_flagged:
            logger.info("All sections approved, finalizing")
        else:
            logger.info(f"Max iterations ({max_iterations}) reached, finalizing")
        return "finalize"


def finalize_node(state: dict[str, Any]) -> dict[str, Any]:
    """Finalize Loop 4 editing."""
    return {"is_complete": True}


def create_loop4_graph() -> StateGraph:
    """Create Loop 4 subgraph with parallel editing flow.

    Flow:
        START → split_sections → parallel_edit_sections → reassemble_document
            → holistic_review → route
                → (sections_flagged AND iteration < max) → split_sections (flagged only)
                → (all_approved OR max_iterations) → finalize → END
    """
    builder = StateGraph(Loop4State)

    # Add nodes
    builder.add_node("split_sections", split_sections_node)
    builder.add_node("parallel_edit_sections", parallel_edit_sections_node)
    builder.add_node("reassemble_document", reassemble_document_node)
    builder.add_node("holistic_review", holistic_review_node)
    builder.add_node("finalize", finalize_node)

    # Entry point
    builder.add_edge(START, "split_sections")

    # Linear flow through editing
    builder.add_edge("split_sections", "parallel_edit_sections")
    builder.add_edge("parallel_edit_sections", "reassemble_document")
    builder.add_edge("reassemble_document", "holistic_review")

    # Route after holistic review
    builder.add_conditional_edges(
        "holistic_review",
        route_after_holistic,
        {
            "split_sections": "split_sections",
            "finalize": "finalize",
        }
    )

    # Exit
    builder.add_edge("finalize", END)

    return builder.compile()


# Export compiled graph
loop4_graph = create_loop4_graph()


async def run_loop4_standalone(
    review: str,
    paper_summaries: dict,
    input_data: LitReviewInput,
    max_iterations: int = 3,
) -> dict:
    """Run Loop 4 as standalone operation for testing.

    Args:
        review: Current literature review text
        paper_summaries: Dictionary of DOI -> PaperSummary
        input_data: Original research input
        max_iterations: Maximum editing iterations

    Returns:
        Dictionary containing:
            - edited_review: Final edited review
            - iterations: Number of iterations performed
            - section_results: Final section editing results
            - holistic_result: Final holistic review result
    """
    initial_state = {
        "current_review": review,
        "paper_summaries": paper_summaries,
        "input": input_data,
        "sections": [],
        "section_results": {},
        "editor_notes": [],
        "holistic_result": None,
        "flagged_sections": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "is_complete": False,
    }

    logger.info(f"Starting Loop 4 standalone: max_iterations={max_iterations}")

    final_state = await loop4_graph.ainvoke(initial_state)

    return {
        "edited_review": final_state.get("current_review", review),
        "iterations": final_state.get("iteration", 0),
        "section_results": final_state.get("section_results", {}),
        "holistic_result": final_state.get("holistic_result"),
    }
