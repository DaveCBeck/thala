"""Loop 4: Section-Level Deep Editing with parallel processing and tool access."""

import asyncio
import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from workflows.academic_lit_review.state import (
    LitReviewInput,
    PaperSummary,
)
from workflows.supervised_lit_review.supervision.types import (
    SectionEditResult,
    HolisticReviewResult,
)
from workflows.supervised_lit_review.supervision.prompts import (
    LOOP4_SECTION_EDITOR_SYSTEM,
    LOOP4_SECTION_EDITOR_USER,
    LOOP4_HOLISTIC_SYSTEM,
    LOOP4_HOLISTIC_USER,
)
from workflows.supervised_lit_review.supervision.utils import (
    split_into_sections,
    SectionInfo,
    format_paper_summaries_with_budget,
    create_manifest_note,
    validate_edit_citations,
    check_section_growth,
)
from workflows.supervised_lit_review.supervision.store_query import (
    SupervisionStoreQuery,
)
from workflows.supervised_lit_review.supervision.tools import (
    create_paper_tools,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


class Loop4State(TypedDict):
    """State schema for Loop 4 section-level editing."""

    current_review: str
    paper_summaries: dict[str, PaperSummary]
    zotero_keys: dict[str, str]  # DOI -> zotero citation key mapping
    input: LitReviewInput

    sections: list[SectionInfo]
    section_results: dict[str, SectionEditResult]
    editor_notes: list[str]

    holistic_result: Optional[HolisticReviewResult]
    flagged_sections: list[str]

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
        sections = split_into_sections(current_review, max_tokens=5000)
        logger.info(f"Split document into {len(sections)} sections for initial editing")
    else:
        all_sections = split_into_sections(current_review, max_tokens=5000)
        sections = [s for s in all_sections if s["section_id"] in flagged]
        logger.info(f"Re-editing {len(sections)} flagged sections")

    return {"sections": sections}


async def parallel_edit_sections_node(state: dict[str, Any]) -> dict[str, Any]:
    """Phase A: Parallel section editing with concurrent Opus calls and dynamic store access."""
    sections = state.get("sections", [])
    paper_summaries = state.get("paper_summaries", {})
    zotero_keys = state.get("zotero_keys", {})

    store_query = SupervisionStoreQuery(paper_summaries)

    semaphore = asyncio.Semaphore(5)

    async def edit_section(section: SectionInfo, section_idx: int) -> tuple[str, SectionEditResult]:
        """Edit a single section with Opus, dynamic store access, and search tools."""
        async with semaphore:
            section_id = section["section_id"]
            section_content = section["section_content"]

            todos = [
                line.strip()
                for line in section_content.split("\n")
                if "<!-- TODO:" in line
            ]

            # Get limited context window (1 section before/after) instead of full document
            context_window = get_section_context_window(sections, section_idx, num_surrounding=1)

            # Filter paper summaries to only papers cited in this section
            cited_papers = get_cited_papers_only(section_content, paper_summaries, zotero_keys)

            # Get detailed content for cited papers only
            detailed_content = await store_query.get_papers_for_section(
                section_content,
                max_papers=5,
                compression_level=2,
                max_total_chars=30000,
            )

            # Format cited papers with budget
            summary_text = format_paper_summaries_with_budget(
                cited_papers,  # Only cited papers
                detailed_content,
                max_total_chars=30000,  # Reduced budget since fewer papers
            )

            manifest_note = create_manifest_note(
                papers_with_detail=len(detailed_content),
                papers_total=len(cited_papers),
                compression_level=2,
            )

            paper_tools = create_paper_tools(paper_summaries, store_query)

            user_prompt = LOOP4_SECTION_EDITOR_USER.format(
                context_window=context_window,
                section_id=section_id,
                section_content=section_content,
                paper_summaries=f"{manifest_note}\n\n{summary_text}",
                todos_in_section="\n".join(todos) if todos else "None"
            )

            response = await get_structured_output(
                output_schema=SectionEditResult,
                user_prompt=user_prompt,
                system_prompt=LOOP4_SECTION_EDITOR_SYSTEM,
                tools=paper_tools,
                tier=ModelTier.OPUS,
                max_tokens=16384,
                max_tool_calls=5,
            )

            # Validate citations - only allow papers already cited in original section
            corpus_keys = set(zotero_keys.values())
            is_valid_citations, invalid_cites = validate_edit_citations(
                original_section=section_content,
                edited_section=response.edited_content,
                corpus_keys=corpus_keys,
            )

            if not is_valid_citations:
                logger.warning(
                    f"Section '{section_id}': Edit rejected - invalid citations: {invalid_cites}"
                )
                response = SectionEditResult(
                    section_id=section_id,
                    edited_content=section_content,  # Keep original
                    notes=f"Edit rejected due to invalid citations: {invalid_cites}",
                    new_paper_todos=response.new_paper_todos,
                    confidence=0.0,
                )
            else:
                # Validate word count - must be within ±20% of original
                is_within_limit, growth = check_section_growth(
                    section_content, response.edited_content, tolerance=0.20
                )

                if not is_within_limit:
                    logger.warning(
                        f"Section '{section_id}': Edit exceeds word limit ({growth:+.1%}), reverting"
                    )
                    response = SectionEditResult(
                        section_id=section_id,
                        edited_content=section_content,  # Keep original
                        notes=f"Edit rejected: word count change {growth:+.1%} exceeds ±20%",
                        new_paper_todos=response.new_paper_todos,
                        confidence=0.0,
                    )

            logger.info(
                f"Edited section '{section_id}' (confidence: {response.confidence:.2f}, "
                f"papers_with_detail: {len(detailed_content)}, tools_available: True)"
            )
            return section_id, response

    edit_tasks = [edit_section(s, idx) for idx, s in enumerate(sections)]
    results = await asyncio.gather(*edit_tasks)

    section_results = {section_id: result for section_id, result in results}

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


def detect_duplicate_sections(sections: list[SectionInfo]) -> list[tuple[str, str]]:
    """Detect sections with overlapping content that may cause duplicates.

    Uses SequenceMatcher to find sections with >70% similarity in their
    first 500 characters, indicating likely duplicates from splitting issues.

    Args:
        sections: List of section info dictionaries

    Returns:
        List of (section_id_1, section_id_2) tuples for duplicate pairs
    """
    from difflib import SequenceMatcher

    duplicates = []
    for i, s1 in enumerate(sections):
        for s2 in sections[i + 1:]:
            # Check for significant overlap in first 500 chars
            ratio = SequenceMatcher(
                None,
                s1["section_content"][:500],
                s2["section_content"][:500]
            ).ratio()
            if ratio > 0.7:
                duplicates.append((s1["section_id"], s2["section_id"]))
    return duplicates


def merge_duplicate_edits(
    section_results: dict[str, SectionEditResult],
    duplicates: list[tuple[str, str]],
) -> dict[str, SectionEditResult]:
    """Merge edits from duplicate sections, keeping the higher-confidence edit.

    When two sections are detected as duplicates, we keep the edit with
    higher confidence and merge their notes and TODOs.

    Args:
        section_results: Dictionary mapping section_id to edit results
        duplicates: List of (section_id_1, section_id_2) duplicate pairs

    Returns:
        Updated section_results with duplicates merged
    """
    merged = dict(section_results)

    for id1, id2 in duplicates:
        if id1 in merged and id2 in merged:
            r1, r2 = merged[id1], merged[id2]
            # Keep higher confidence edit, merge notes
            if r2.confidence > r1.confidence:
                merged[id1] = SectionEditResult(
                    section_id=id1,
                    edited_content=r2.edited_content,
                    notes=f"{r1.notes}\n{r2.notes}".strip(),
                    new_paper_todos=r1.new_paper_todos + r2.new_paper_todos,
                    confidence=r2.confidence,
                )
            else:
                # Keep r1, just merge notes from r2
                merged[id1] = SectionEditResult(
                    section_id=id1,
                    edited_content=r1.edited_content,
                    notes=f"{r1.notes}\n{r2.notes}".strip(),
                    new_paper_todos=r1.new_paper_todos + r2.new_paper_todos,
                    confidence=r1.confidence,
                )
            del merged[id2]  # Remove duplicate

    return merged


def get_section_context_window(
    sections: list[SectionInfo],
    current_idx: int,
    num_surrounding: int = 1,
) -> str:
    """Get limited context: current section + surrounding sections.

    Instead of passing the full document to section editors, we provide
    just the surrounding sections for context. This dramatically reduces
    token usage while preserving the context needed for coherent editing.

    Args:
        sections: List of all section info dictionaries
        current_idx: Index of the current section being edited
        num_surrounding: Number of sections before/after to include

    Returns:
        Formatted context string with surrounding sections
    """
    start_idx = max(0, current_idx - num_surrounding)
    end_idx = min(len(sections), current_idx + num_surrounding + 1)

    context_parts = []
    for i in range(start_idx, end_idx):
        prefix = ">>> CURRENT SECTION <<<\n" if i == current_idx else ""
        context_parts.append(f"{prefix}{sections[i]['section_content']}")

    return "\n\n---\n\n".join(context_parts)


def get_cited_papers_only(
    section_content: str,
    paper_summaries: dict[str, PaperSummary],
    zotero_keys: dict[str, str],
) -> dict[str, PaperSummary]:
    """Filter paper summaries to only those cited in section.

    Instead of passing all paper summaries to section editors, we only
    provide summaries for papers that are actually cited in the section.
    This reduces token usage and helps enforce the corpus-only citation policy.

    Args:
        section_content: The section text to search for citations
        paper_summaries: Full dictionary of DOI -> PaperSummary
        zotero_keys: Dictionary of DOI -> zotero citation key

    Returns:
        Filtered dictionary containing only cited papers
    """
    from ..utils.citation_validation import extract_citation_keys_from_text

    cited_keys = extract_citation_keys_from_text(section_content)

    # Map zotero keys back to DOIs
    key_to_doi = {v: k for k, v in zotero_keys.items()}
    cited_dois = {key_to_doi.get(key) for key in cited_keys if key in key_to_doi}

    return {doi: paper_summaries[doi] for doi in cited_dois if doi in paper_summaries}


def reassemble_document_node(state: dict[str, Any]) -> dict[str, Any]:
    """Reassemble document from edited sections."""
    current_review = state.get("current_review", "")
    sections = state.get("sections", [])
    section_results = state.get("section_results", {})
    iteration = state.get("iteration", 0)

    # Detect and merge duplicate sections before reassembly
    duplicates = detect_duplicate_sections(sections)
    if duplicates:
        logger.info(f"Detected {len(duplicates)} duplicate section pair(s): {duplicates}")
        section_results = merge_duplicate_edits(section_results, duplicates)

    if iteration == 0:
        edited_content = []
        for section in sections:
            section_id = section["section_id"]
            if section_id in section_results:
                edited_content.append(section_results[section_id].edited_content)
            else:
                edited_content.append(section["section_content"])

        updated_review = "\n\n".join(edited_content)
    else:
        lines = current_review.split("\n")

        sorted_sections = sorted(
            [s for s in sections if s["section_id"] in section_results],
            key=lambda s: s["start_line"],
            reverse=True
        )

        for section in sorted_sections:
            section_id = section["section_id"]
            start_line = section["start_line"]
            end_line = section["end_line"]
            edited_lines = section_results[section_id].edited_content.split("\n")

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

    user_prompt = LOOP4_HOLISTIC_USER.format(
        document=document,
        editor_notes="\n".join(editor_notes),
        iteration=iteration + 1,
        max_iterations=max_iterations
    )

    try:
        result = await get_structured_output(
            output_schema=HolisticReviewResult,
            user_prompt=user_prompt,
            system_prompt=LOOP4_HOLISTIC_SYSTEM,
            tier=ModelTier.OPUS,
            max_tokens=8192,
            use_json_schema_method=True,
            max_retries=2,
        )

        logger.info(
            f"Holistic review: {len(result.sections_approved)} approved, "
            f"{len(result.sections_flagged)} flagged (coherence: {result.overall_coherence_score:.2f})"
        )

    except Exception as e:
        logger.error(f"Holistic review failed: {e}")
        result = HolisticReviewResult(
            sections_approved=[],
            sections_flagged=[],
            flagged_reasons={},
            overall_coherence_score=0.5,
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

    builder.add_node("split_sections", split_sections_node)
    builder.add_node("parallel_edit_sections", parallel_edit_sections_node)
    builder.add_node("reassemble_document", reassemble_document_node)
    builder.add_node("holistic_review", holistic_review_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "split_sections")

    builder.add_edge("split_sections", "parallel_edit_sections")
    builder.add_edge("parallel_edit_sections", "reassemble_document")
    builder.add_edge("reassemble_document", "holistic_review")

    builder.add_conditional_edges(
        "holistic_review",
        route_after_holistic,
        {
            "split_sections": "split_sections",
            "finalize": "finalize",
        }
    )

    builder.add_edge("finalize", END)

    return builder.compile()


loop4_graph = create_loop4_graph()


async def run_loop4_standalone(
    review: str,
    paper_summaries: dict,
    input_data: LitReviewInput,
    zotero_keys: dict[str, str],
    max_iterations: int = 3,
    config: dict | None = None,
) -> dict:
    """Run Loop 4 as standalone operation for testing.

    Args:
        review: Current literature review text
        paper_summaries: Dictionary of DOI -> PaperSummary
        input_data: Original research input
        zotero_keys: Dictionary of DOI -> zotero citation key for validation
        max_iterations: Maximum editing iterations
        config: Optional LangGraph config with run_id and run_name for tracing

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
        "zotero_keys": zotero_keys,
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

    if config:
        final_state = await loop4_graph.ainvoke(initial_state, config=config)
    else:
        final_state = await loop4_graph.ainvoke(initial_state)

    return {
        "edited_review": final_state.get("current_review", review),
        "iterations": final_state.get("iteration", 0),
        "section_results": final_state.get("section_results", {}),
        "holistic_result": final_state.get("holistic_result"),
    }
