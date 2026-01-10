"""Loop 4: Section-Level Deep Editing with parallel processing and tool access."""

import asyncio
import logging
import re
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from core.stores.zotero import ZoteroStore
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
    validate_edit_citations_with_zotero,
    strip_invalid_citations,
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

    # Zotero verification settings
    verify_zotero: bool
    verified_citation_keys: set[str]


def detect_duplicate_sections(sections: list[SectionInfo]) -> list[tuple[str, str]]:
    """Detect sections with overlapping content that may cause duplicates."""
    from difflib import SequenceMatcher

    duplicates = []
    for i, s1 in enumerate(sections):
        for s2 in sections[i + 1:]:
            ratio = SequenceMatcher(
                None,
                s1["section_content"][:500],
                s2["section_content"][:500]
            ).ratio()
            if ratio > 0.7:
                duplicates.append((s1["section_id"], s2["section_id"]))
    return duplicates


def detect_duplicate_headers(document: str) -> list[tuple[int, int, str]]:
    """Detect duplicate section headers in document."""
    lines = document.split("\n")
    header_positions: dict[str, list[int]] = {}

    for i, line in enumerate(lines):
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if header_match:
            header_text = header_match.group(0).strip().lower()
            if header_text not in header_positions:
                header_positions[header_text] = []
            header_positions[header_text].append(i)

    duplicates = []
    for header_text, positions in header_positions.items():
        if len(positions) > 1:
            for i in range(len(positions) - 1):
                duplicates.append((positions[i], positions[i + 1], header_text))
                logger.warning(
                    f"Duplicate header found: '{header_text}' at lines {positions[i] + 1} and {positions[i + 1] + 1}"
                )

    return duplicates


def remove_duplicate_headers(document: str, duplicates: list[tuple[int, int, str]]) -> str:
    """Remove duplicate section headers from document."""
    from difflib import SequenceMatcher

    if not duplicates:
        return document

    lines = document.split("\n")
    sorted_dups = sorted(duplicates, key=lambda x: x[1], reverse=True)

    for line1, line2, header_text in sorted_dups:
        end_line = line2

        for i in range(line2 + 1, len(lines)):
            header_match = re.match(r"^(#{1,6})\s+(.+)$", lines[i].strip())
            if header_match:
                break
            end_line = i

        content_start1 = line1 + 1
        content_end1 = line2 - 1
        content_start2 = line2 + 1
        content_end2 = end_line

        content1 = "\n".join(lines[content_start1:content_end1 + 1]).strip()[:500]
        content2 = "\n".join(lines[content_start2:content_end2 + 1]).strip()[:500]

        similarity = SequenceMatcher(None, content1, content2).ratio()

        if similarity > 0.5:
            logger.info(
                f"Removing duplicate section at lines {line2 + 1}-{end_line + 1} "
                f"(similarity: {similarity:.2f})"
            )
            lines = lines[:line2] + lines[end_line + 1:]
        else:
            logger.info(
                f"Removing only duplicate header at line {line2 + 1} "
                f"(content similarity too low: {similarity:.2f})"
            )
            lines = lines[:line2] + lines[line2 + 1:]

    return "\n".join(lines)


def merge_duplicate_edits(
    section_results: dict[str, SectionEditResult],
    duplicates: list[tuple[str, str]],
) -> dict[str, SectionEditResult]:
    """Merge edits from duplicate sections, keeping the higher-confidence edit."""
    merged = dict(section_results)

    for id1, id2 in duplicates:
        if id1 in merged and id2 in merged:
            r1, r2 = merged[id1], merged[id2]
            if r2.confidence > r1.confidence:
                merged[id1] = SectionEditResult(
                    section_id=id1,
                    edited_content=r2.edited_content,
                    notes=f"{r1.notes}\n{r2.notes}".strip(),
                    new_paper_todos=r1.new_paper_todos + r2.new_paper_todos,
                    confidence=r2.confidence,
                )
            else:
                merged[id1] = SectionEditResult(
                    section_id=id1,
                    edited_content=r1.edited_content,
                    notes=f"{r1.notes}\n{r2.notes}".strip(),
                    new_paper_todos=r1.new_paper_todos + r2.new_paper_todos,
                    confidence=r1.confidence,
                )
            del merged[id2]

    return merged


def get_section_context_window(
    sections: list[SectionInfo],
    current_idx: int,
    num_surrounding: int = 1,
) -> str:
    """Get limited context: current section + surrounding sections."""
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
    """Filter paper summaries to only those cited in section."""
    from ..utils.citation_validation import extract_citation_keys_from_text

    cited_keys = extract_citation_keys_from_text(section_content)
    key_to_doi = {v: k for k, v in zotero_keys.items()}
    cited_dois = {key_to_doi.get(key) for key in cited_keys if key in key_to_doi}

    return {doi: paper_summaries[doi] for doi in cited_dois if doi in paper_summaries}


def _log_section_info(stage: str, sections: list[SectionInfo]) -> None:
    """Log section headers and IDs for debugging duplicate issues."""
    section_ids = [s["section_id"] for s in sections]
    logger.debug(f"[{stage}] Section count: {len(sections)}")
    logger.debug(f"[{stage}] Section IDs: {section_ids}")

    seen_ids = set()
    for sid in section_ids:
        if sid in seen_ids:
            logger.warning(f"[{stage}] Duplicate section_id detected: {sid}")
        seen_ids.add(sid)


def _format_section_id_list(sections: list[SectionInfo]) -> str:
    """Format section ID list for the holistic reviewer prompt."""
    lines = []
    for s in sections:
        level_indicator = "#" * s["heading_level"] if s["heading_level"] > 0 else "-"
        lines.append(f"- `{s['section_id']}` ({level_indicator})")
    return "\n".join(lines)


def split_sections_node(state: dict[str, Any]) -> dict[str, Any]:
    """Split document into sections for parallel editing."""
    current_review = state.get("current_review", "")
    iteration = state.get("iteration", 0)
    flagged = state.get("flagged_sections", [])

    logger.debug(f"[split_sections] Input document length: {len(current_review)} chars")

    if iteration == 0:
        sections = split_into_sections(current_review, max_tokens=5000)
        logger.info(f"Split document into {len(sections)} sections for initial editing")
    else:
        all_sections = split_into_sections(current_review, max_tokens=5000)
        sections = [s for s in all_sections if s["section_id"] in flagged]
        logger.info(f"Re-editing {len(sections)} flagged sections")

    _log_section_info("split_sections", sections)

    return {"sections": sections}


async def parallel_edit_sections_node(state: dict[str, Any]) -> dict[str, Any]:
    """Phase A: Parallel section editing with concurrent Opus calls and dynamic store access."""
    sections = state.get("sections", [])
    paper_summaries = state.get("paper_summaries", {})
    zotero_keys = state.get("zotero_keys", {})
    verify_zotero = state.get("verify_zotero", False)
    verified_keys = state.get("verified_citation_keys", set())

    _log_section_info("parallel_edit_input", sections)

    store_query = SupervisionStoreQuery(paper_summaries)

    # Initialize Zotero client if verification is enabled
    zotero_client: Optional[ZoteroStore] = None
    if verify_zotero:
        zotero_client = ZoteroStore()

    semaphore = asyncio.Semaphore(5)

    async def edit_section(
        section: SectionInfo, section_idx: int
    ) -> tuple[str, SectionEditResult, set[str]]:
        """Edit a single section with Opus, dynamic store access, and search tools."""
        async with semaphore:
            section_id = section["section_id"]
            section_content = section["section_content"]
            newly_verified: set[str] = set()

            original_word_count = len(section_content.split())
            if original_word_count < 50:
                section_category = "very_short"
                tolerance = None
            elif original_word_count < 150:
                section_category = "short"
                tolerance = 0.50
            else:
                section_category = "normal"
                tolerance = 0.20

            todos = [
                line.strip()
                for line in section_content.split("\n")
                if "<!-- TODO:" in line
            ]

            context_window = get_section_context_window(sections, section_idx, num_surrounding=1)
            cited_papers = get_cited_papers_only(section_content, paper_summaries, zotero_keys)

            detailed_content = await store_query.get_papers_for_section(
                section_content,
                max_papers=5,
                compression_level=2,
                max_total_chars=30000,
            )

            summary_text = format_paper_summaries_with_budget(
                cited_papers,
                detailed_content,
                max_total_chars=30000,
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

            corpus_keys = set(zotero_keys.values())

            # Citation validation - with optional Zotero verification
            if verify_zotero and zotero_client:
                is_valid, invalid_cites, new_verified = await validate_edit_citations_with_zotero(
                    original_section=section_content,
                    edited_section=response.edited_content,
                    corpus_keys=corpus_keys | verified_keys,
                    zotero_client=zotero_client,
                )
                newly_verified = new_verified - corpus_keys - verified_keys

                if not is_valid:
                    invalid_key_set = {c.split(" ")[0] for c in invalid_cites}
                    cleaned_content = strip_invalid_citations(
                        response.edited_content, invalid_key_set, add_todo=True
                    )
                    logger.warning(
                        f"Section '{section_id}': Stripped {len(invalid_cites)} unverified citations"
                    )
                    response = SectionEditResult(
                        section_id=section_id,
                        edited_content=cleaned_content,
                        notes=f"Stripped unverified citations: {invalid_cites}",
                        new_paper_todos=response.new_paper_todos,
                        confidence=response.confidence * 0.9,
                    )
            else:
                is_valid, invalid_cites = validate_edit_citations(
                    original_section=section_content,
                    edited_section=response.edited_content,
                    corpus_keys=corpus_keys,
                )

                if not is_valid:
                    logger.warning(
                        f"Section '{section_id}': Edit rejected - invalid citations: {invalid_cites}"
                    )
                    response = SectionEditResult(
                        section_id=section_id,
                        edited_content=section_content,
                        notes=f"Edit rejected due to invalid citations: {invalid_cites}",
                        new_paper_todos=response.new_paper_todos,
                        confidence=0.0,
                    )

            # Word count validation
            if response.confidence > 0 and tolerance is not None:
                is_within_limit, growth = check_section_growth(
                    section_content, response.edited_content, tolerance=tolerance
                )

                if not is_within_limit:
                    logger.warning(
                        f"Section '{section_id}' ({section_category}): word count {growth:+.1%} exceeds "
                        f"+/-{tolerance*100:.0f}%, retrying with compression"
                    )

                    edited_word_count = len(response.edited_content.split())
                    max_allowed = int(original_word_count * (1 + tolerance))

                    retry_prompt = f"""Your previous edit exceeded the word limit.

Original section: {original_word_count} words
Your edit: {edited_word_count} words
Maximum allowed: {max_allowed} words (+/-{tolerance*100:.0f}% tolerance)

Please revise your edit to fit within the limit. You MUST compress or remove content to stay within bounds.
Keep the most important improvements and cut the rest.
Return the section unchanged if you cannot make meaningful improvements within the limit.

## Original Section
{section_content}

## Your Previous Edit (too long)
{response.edited_content}"""

                    retry_response = await get_structured_output(
                        output_schema=SectionEditResult,
                        user_prompt=retry_prompt,
                        system_prompt="You are revising a section edit to fit within word count limits. Compress the content while preserving the key improvements.",
                        tier=ModelTier.OPUS,
                        max_tokens=16384,
                    )

                    # Validate retry citations
                    if verify_zotero and zotero_client:
                        is_valid_retry, retry_invalid, retry_verified = await validate_edit_citations_with_zotero(
                            original_section=section_content,
                            edited_section=retry_response.edited_content,
                            corpus_keys=corpus_keys | verified_keys | newly_verified,
                            zotero_client=zotero_client,
                        )
                        newly_verified |= retry_verified - corpus_keys - verified_keys
                    else:
                        is_valid_retry, retry_invalid = validate_edit_citations(
                            original_section=section_content,
                            edited_section=retry_response.edited_content,
                            corpus_keys=corpus_keys,
                        )

                    if not is_valid_retry:
                        logger.warning(
                            f"Section '{section_id}': Retry also has invalid citations: {retry_invalid}"
                        )
                        response = SectionEditResult(
                            section_id=section_id,
                            edited_content=section_content,
                            notes=f"Edit rejected after retry - invalid citations: {retry_invalid}",
                            new_paper_todos=response.new_paper_todos,
                            confidence=0.0,
                        )
                    else:
                        is_retry_within_limit, retry_growth = check_section_growth(
                            section_content, retry_response.edited_content, tolerance=tolerance
                        )

                        if is_retry_within_limit:
                            logger.info(
                                f"Section '{section_id}': Retry succeeded ({retry_growth:+.1%})"
                            )
                            response = retry_response
                        else:
                            logger.warning(
                                f"Section '{section_id}': Retry still exceeds limit ({retry_growth:+.1%}), reverting"
                            )
                            response = SectionEditResult(
                                section_id=section_id,
                                edited_content=section_content,
                                notes=f"Edit rejected after retry: word count {retry_growth:+.1%} still exceeds +/-{tolerance*100:.0f}%",
                                new_paper_todos=response.new_paper_todos,
                                confidence=0.0,
                            )
            elif response.confidence > 0 and tolerance is None:
                logger.info(
                    f"Section '{section_id}' ({section_category}, {original_word_count} words): "
                    f"no word limit applied"
                )

            logger.info(
                f"Edited section '{section_id}' (confidence: {response.confidence:.2f}, "
                f"category: {section_category}, papers_with_detail: {len(detailed_content)})"
            )
            return section_id, response, newly_verified

    edit_tasks = [edit_section(s, idx) for idx, s in enumerate(sections)]
    results = await asyncio.gather(*edit_tasks)

    # Close Zotero client
    if zotero_client:
        await zotero_client.close()

    section_results = {section_id: result for section_id, result, _ in results}

    # Collect all newly verified keys
    all_newly_verified: set[str] = set()
    for _, _, newly_verified in results:
        all_newly_verified |= newly_verified

    logger.debug(f"[parallel_edit_output] Edited section IDs: {list(section_results.keys())}")

    editor_notes = [
        f"[{section_id}] {result.notes}"
        for section_id, result in section_results.items()
        if result.notes
    ]

    logger.info(f"Completed parallel editing of {len(section_results)} sections")
    if all_newly_verified:
        logger.info(f"Verified {len(all_newly_verified)} new citation keys against Zotero")

    return {
        "section_results": section_results,
        "editor_notes": editor_notes,
        "verified_citation_keys": verified_keys | all_newly_verified,
    }


def reassemble_document_node(state: dict[str, Any]) -> dict[str, Any]:
    """Reassemble document from edited sections."""
    current_review = state.get("current_review", "")
    sections = state.get("sections", [])
    section_results = state.get("section_results", {})
    iteration = state.get("iteration", 0)

    _log_section_info("reassemble_input", sections)
    logger.debug(f"[reassemble] Section results keys: {list(section_results.keys())}")

    duplicates = detect_duplicate_sections(sections)
    if duplicates:
        logger.info(f"Detected {len(duplicates)} duplicate section pair(s): {duplicates}")
        section_results = merge_duplicate_edits(section_results, duplicates)
        logger.debug(f"[reassemble] After merge, section results keys: {list(section_results.keys())}")

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

            logger.debug(
                f"[reassemble] Replacing section '{section_id}' at lines {start_line}-{end_line} "
                f"with {len(edited_lines)} lines"
            )

            lines = lines[:start_line] + edited_lines + lines[end_line + 1:]

        updated_review = "\n".join(lines)

    logger.debug(f"[reassemble_output] Document length: {len(updated_review)} chars")
    logger.info(f"Reassembled document: {len(updated_review)} chars")

    return {"current_review": updated_review}


async def holistic_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """Phase B: Holistic coherence review with single Opus call."""
    document = state.get("current_review", "")
    editor_notes = state.get("editor_notes", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    all_sections = split_into_sections(document, max_tokens=5000)
    section_ids = [s["section_id"] for s in all_sections]
    section_id_list = _format_section_id_list(all_sections)

    logger.info(f"Holistic review: evaluating {len(section_ids)} sections")
    logger.debug(f"[holistic_review] Section IDs passed to LLM: {section_ids}")

    user_prompt = LOOP4_HOLISTIC_USER.format(
        document=document,
        editor_notes="\n".join(editor_notes),
        iteration=iteration + 1,
        max_iterations=max_iterations,
        section_id_list=section_id_list,
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

        logger.debug(
            f"[holistic_review] Raw LLM response: approved={result.sections_approved}, "
            f"flagged={result.sections_flagged}, coherence={result.overall_coherence_score}"
        )

        valid_ids = set(section_ids)
        approved_valid = [sid for sid in result.sections_approved if sid in valid_ids]
        flagged_valid = [sid for sid in result.sections_flagged if sid in valid_ids]

        if len(approved_valid) != len(result.sections_approved):
            invalid_approved = set(result.sections_approved) - valid_ids
            logger.warning(f"[holistic_review] Invalid approved IDs (filtered out): {invalid_approved}")

        if len(flagged_valid) != len(result.sections_flagged):
            invalid_flagged = set(result.sections_flagged) - valid_ids
            logger.warning(f"[holistic_review] Invalid flagged IDs (filtered out): {invalid_flagged}")

        result = HolisticReviewResult(
            sections_approved=approved_valid,
            sections_flagged=flagged_valid,
            flagged_reasons={k: v for k, v in result.flagged_reasons.items() if k in valid_ids},
            overall_coherence_score=result.overall_coherence_score,
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

    if not has_flagged and holistic.overall_coherence_score < 0.7:
        logger.warning(
            f"Low coherence score ({holistic.overall_coherence_score:.2f}) but no sections flagged - "
            "holistic review may need debugging"
        )

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
    """Finalize Loop 4 editing with duplicate cleanup."""
    document = state.get("current_review", "")

    logger.debug(f"[finalize] Input document length: {len(document)} chars")

    duplicates = detect_duplicate_headers(document)
    if duplicates:
        logger.warning(f"[finalize] Found {len(duplicates)} duplicate header(s), cleaning up")
        document = remove_duplicate_headers(document, duplicates)
        logger.info(f"[finalize] Cleaned document length: {len(document)} chars")

        remaining_duplicates = detect_duplicate_headers(document)
        if remaining_duplicates:
            logger.error(
                f"[finalize] Still have {len(remaining_duplicates)} duplicate(s) after cleanup"
            )
    else:
        logger.debug("[finalize] No duplicate headers detected")

    return {"current_review": document, "is_complete": True}


def create_loop4_graph() -> StateGraph:
    """Create Loop 4 subgraph with parallel editing flow."""
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
    verify_zotero: bool = False,
) -> dict:
    """Run Loop 4 as standalone operation for testing.

    Args:
        review: Current literature review text
        paper_summaries: Dictionary of DOI -> PaperSummary
        input_data: Original research input
        zotero_keys: Dictionary of DOI -> zotero citation key for validation
        max_iterations: Maximum editing iterations
        config: Optional LangGraph config with run_id and run_name for tracing
        verify_zotero: If True, verify new citations against Zotero programmatically

    Returns:
        Dictionary containing edited_review, iterations, section_results, holistic_result,
        and verified_citation_keys (if verify_zotero=True)
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
        "verify_zotero": verify_zotero,
        "verified_citation_keys": set(),
    }

    logger.info(f"Starting Loop 4 standalone: max_iterations={max_iterations}, verify_zotero={verify_zotero}")

    if config:
        final_state = await loop4_graph.ainvoke(initial_state, config=config)
    else:
        final_state = await loop4_graph.ainvoke(initial_state)

    return {
        "edited_review": final_state.get("current_review", review),
        "iterations": final_state.get("iteration", 0),
        "section_results": final_state.get("section_results", {}),
        "holistic_result": final_state.get("holistic_result"),
        "verified_citation_keys": final_state.get("verified_citation_keys", set()),
    }
