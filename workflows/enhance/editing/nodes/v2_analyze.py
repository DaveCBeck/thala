"""V2 Phase 1: Global analysis node.

This node analyzes the full document and identifies sections
that need structural improvements.
"""

import logging
from typing import Any

from langsmith import traceable

from workflows.shared.llm_utils import ModelTier, get_structured_output

from ..prompts import V2_GLOBAL_ANALYSIS_SYSTEM, V2_GLOBAL_ANALYSIS_USER
from ..schemas import GlobalAnalysisResult, TopLevelSection, parse_sections

logger = logging.getLogger(__name__)


def build_sections_summary(sections: list[TopLevelSection]) -> str:
    """Build a summary of sections for the LLM prompt.

    Args:
        sections: List of parsed sections

    Returns:
        Formatted summary string
    """
    lines = []
    for section in sections:
        citations_str = f" (citations: {len(section.citations)})" if section.citations else ""
        lines.append(
            f"[{section.index}] # {section.heading} ({section.word_count} words){citations_str}"
        )
    return "\n".join(lines)


@traceable(run_type="chain", name="EditingV2.Analyze")
async def v2_analyze_node(state: dict) -> dict[str, Any]:
    """V2 Phase 1: Analyze document structure and identify sections needing work.

    This node:
    1. Parses the document into top-level sections
    2. Sends the full document to the LLM for analysis
    3. Returns a list of edit instructions for problematic sections

    Args:
        state: Current workflow state

    Returns:
        State update with sections and edit_instructions
    """
    # Extract inputs
    input_data = state["input"]
    document = input_data["document"]
    topic = input_data["topic"]
    quality_settings = state.get("quality_settings", {})

    # Parse document into sections
    sections = parse_sections(document)
    logger.info(f"Parsed document into {len(sections)} top-level sections")

    if len(sections) == 0:
        logger.warning("No sections found in document")
        return {
            "sections": [],
            "edit_instructions": [],
            "analysis_complete": True,
            "errors": [{"phase": "analyze", "error": "No sections found in document"}],
        }

    # Build prompt
    sections_summary = build_sections_summary(sections)

    user_prompt = V2_GLOBAL_ANALYSIS_USER.format(
        topic=topic,
        document=document,
        sections_summary=sections_summary,
    )

    # Determine model tier from quality settings
    use_opus = quality_settings.get("use_opus_for_analysis", True)
    thinking_budget = quality_settings.get("analysis_thinking_budget", 6000)

    # Call LLM for analysis
    try:
        analysis = await get_structured_output(
            output_schema=GlobalAnalysisResult,
            user_prompt=user_prompt,
            system_prompt=V2_GLOBAL_ANALYSIS_SYSTEM,
            tier=ModelTier.OPUS if use_opus else ModelTier.SONNET,
            thinking_budget=thinking_budget if use_opus else None,
            max_tokens=8000,
            use_json_schema_method=True,
            max_retries=2,
        )
    except Exception as e:
        logger.error(f"Analysis LLM call failed: {e}")
        return {
            "sections": [s.model_dump() for s in sections],
            "edit_instructions": [],
            "analysis_complete": True,
            "errors": [{"phase": "analyze", "error": str(e)}],
        }

    # Validate edit instructions reference valid section indices
    # Also prevent duplicate or conflicting instructions for the same section
    valid_instructions = []
    section_indices = {s.index for s in sections}
    seen_sections: set[int] = set()  # Track which sections already have instructions

    for instruction in analysis.instructions:
        if instruction.section_index not in section_indices:
            logger.warning(
                f"Instruction references invalid section index: {instruction.section_index}"
            )
            continue

        # Prevent duplicate instructions for same section
        if instruction.section_index in seen_sections:
            logger.warning(
                f"Duplicate instruction for section {instruction.section_index}, "
                f"skipping: {instruction.instruction_type}"
            )
            continue

        if instruction.instruction_type == "merge_into":
            if (
                instruction.merge_source_index is None
                or instruction.merge_source_index not in section_indices
            ):
                logger.warning(
                    f"Merge instruction has invalid merge_source_index: {instruction.merge_source_index}"
                )
                continue
            # Also mark the source section as having an instruction (it will be removed)
            seen_sections.add(instruction.merge_source_index)

        valid_instructions.append(instruction)
        seen_sections.add(instruction.section_index)

    logger.info(
        f"Analysis complete: {analysis.overall_assessment[:100]}... "
        f"({len(valid_instructions)} instructions)"
    )

    # Log instruction summary
    for instr in valid_instructions:
        section = sections[instr.section_index]
        logger.info(f"  [{instr.section_index}] {section.heading}: {instr.instruction_type}")

    return {
        "sections": [s.model_dump() for s in sections],
        "edit_instructions": [i.model_dump() for i in valid_instructions],
        "analysis_complete": True,
    }
