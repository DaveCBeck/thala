"""V2 Phase 2: Section rewriting node.

This node rewrites a single section following the edit instruction.
It runs in parallel for each section that needs work.
"""

import logging
from typing import Any

from langsmith import traceable

from workflows.shared.llm_utils import ModelTier, get_llm

from ..prompts import V2_SECTION_REWRITE_SYSTEM, V2_SECTION_REWRITE_USER, V2_SECTION_MERGE_USER
from ..schemas import (
    EditInstruction,
    RewrittenSection,
    SectionValidation,
    TopLevelSection,
    extract_v2_citations,
)

logger = logging.getLogger(__name__)


def get_context_window(
    sections: list[TopLevelSection], index: int, position: str, max_words: int = 500
) -> str:
    """Get context from adjacent sections.

    Args:
        sections: All document sections
        index: Current section index
        position: "before" or "after"
        max_words: Maximum words to include

    Returns:
        Context string (ending or beginning of adjacent section)
    """
    if position == "before":
        if index == 0:
            return "(This is the first section - no preceding content)"
        prev_section = sections[index - 1]
        words = prev_section.full_content.split()
        if len(words) <= max_words:
            return prev_section.full_content
        return "...\n" + " ".join(words[-max_words:])

    elif position == "after":
        if index >= len(sections) - 1:
            return "(This is the last section - no following content)"
        next_section = sections[index + 1]
        words = next_section.full_content.split()
        if len(words) <= max_words:
            return next_section.full_content
        return " ".join(words[:max_words]) + "\n..."

    return ""


def validate_rewrite(
    original: TopLevelSection,
    rewritten_content: str,
    instruction_type: str = "rewrite",
    length_tolerance: float = 0.3,
) -> SectionValidation:
    """Validate a rewritten section.

    Checks:
    - Length within tolerance of original (adjusted by instruction type)
    - All original citations preserved

    Args:
        original: Original section
        rewritten_content: Rewritten content
        instruction_type: Type of edit (rewrite, expand, condense, merge_into)
        length_tolerance: Base allowed length deviation (0.3 = ±30%)

    Returns:
        SectionValidation with results
    """
    original_words = original.word_count
    rewritten_words = len(rewritten_content.split())

    original_citations = original.citations
    rewritten_citations = extract_v2_citations(rewritten_content)

    # Adjust tolerance based on instruction type
    # Warning thresholds (soft limits - just log a warning)
    # Failure thresholds (hard limits - fail validation)
    if instruction_type == "expand":
        # Allow up to 100% expansion (2x original) and 10% reduction
        warn_min_ratio = 0.9
        warn_max_ratio = 2.0
        fail_max_ratio = 3.0  # Only fail if > 3x expansion
    elif instruction_type == "condense":
        # Condense: allow any reduction, warn if very aggressive
        warn_min_ratio = 0.4  # Warn below 40%
        warn_max_ratio = 1.1
        fail_max_ratio = 1.5  # Fail if it expands instead of condensing
    elif instruction_type == "merge_into":
        # For merges, allow significant reduction (consolidation)
        warn_min_ratio = 0.4
        warn_max_ratio = 1.1
        fail_max_ratio = 1.5
    else:
        # Standard rewrite: use default tolerance
        warn_min_ratio = 1 - length_tolerance
        warn_max_ratio = 1 + length_tolerance
        fail_max_ratio = 2.0

    # Calculate ratio
    ratio = rewritten_words / original_words if original_words > 0 else 1.0

    # Check for warnings (soft limits)
    length_warning = None
    if original_words > 0:
        if ratio < warn_min_ratio:
            length_warning = (
                f"Aggressive reduction: {rewritten_words} words "
                f"({ratio:.1%} of original {original_words})"
            )
        elif ratio > warn_max_ratio:
            length_warning = (
                f"Significant expansion: {rewritten_words} words "
                f"({ratio:.1%} of original {original_words})"
            )

    # Check for failures (hard limits) - only fail on extreme expansion
    length_ok = ratio <= fail_max_ratio

    # Check citations preserved
    citations_ok = set(original_citations).issubset(set(rewritten_citations))

    # Determine validation result
    passes = length_ok and citations_ok
    rejection_reason = None

    if not length_ok:
        rejection_reason = (
            f"Extreme length change: {rewritten_words} words "
            f"({ratio:.1%} of original {original_words}, "
            f"max allowed {fail_max_ratio:.0%})"
        )
    elif not citations_ok:
        missing = set(original_citations) - set(rewritten_citations)
        rejection_reason = f"Missing citations: {', '.join(missing)}"

    return SectionValidation(
        original_word_count=original_words,
        rewritten_word_count=rewritten_words,
        original_citations=original_citations,
        rewritten_citations=rewritten_citations,
        passes_validation=passes,
        rejection_reason=rejection_reason,
        length_warning=length_warning,
    )


@traceable(run_type="chain", name="EditingV2.RewriteSection")
async def v2_rewrite_section_node(state: dict) -> dict[str, Any]:
    """Rewrite a single section following the edit instruction.

    This node is called via Send() for parallel execution.
    The state passed in contains:
    - sections: All document sections (for context)
    - instruction: The specific edit instruction
    - topic: Document topic
    - quality_settings: Quality tier settings

    Args:
        state: Worker state with section and instruction

    Returns:
        State update with rewritten_sections (single item list for accumulation)
    """
    # Extract from worker state
    sections_data = state["sections"]
    instruction_data = state["instruction"]
    topic = state["topic"]
    quality_settings = state.get("quality_settings", {})

    # Reconstruct objects
    sections = [TopLevelSection(**s) for s in sections_data]
    instruction = EditInstruction(**instruction_data)

    section_index = instruction.section_index
    section = sections[section_index]

    logger.info(
        f"Rewriting section [{section_index}] '{section.heading}' "
        f"({instruction.instruction_type})"
    )

    # Handle delete instruction - no LLM call needed
    if instruction.instruction_type == "delete":
        logger.info(f"Section [{section_index}] marked for deletion")
        return {
            "rewritten_sections": [
                RewrittenSection(
                    section_index=section_index,
                    instruction_type="delete",
                    original_heading=section.heading,
                    new_content="",  # Empty content signals deletion
                    validation=SectionValidation(
                        original_word_count=section.word_count,
                        rewritten_word_count=0,
                        original_citations=section.citations,
                        rewritten_citations=[],
                        passes_validation=True,
                        rejection_reason=None,
                    ),
                ).model_dump()
            ]
        }

    # Get context from adjacent sections
    prev_context = get_context_window(sections, section_index, "before")
    next_context = get_context_window(sections, section_index, "after")

    # Calculate target word count
    original_words = section.word_count
    tolerance = quality_settings.get("enhance_word_tolerance", 0.3)
    target_min = int(original_words * (1 - tolerance))
    target_max = int(original_words * (1 + tolerance))

    # Adjust targets based on instruction type
    if instruction.instruction_type == "expand":
        target_max = int(original_words * 1.5)  # Allow 50% expansion
    elif instruction.instruction_type == "condense":
        target_min = int(original_words * 0.5)  # Allow 50% reduction

    # Build prompt based on instruction type
    if instruction.instruction_type == "merge_into" and instruction.merge_source_index is not None:
        # Merge instruction - combine two sections
        source_section = sections[instruction.merge_source_index]
        combined_words = section.word_count + source_section.word_count
        all_citations = list(set(section.citations + source_section.citations))

        user_prompt = V2_SECTION_MERGE_USER.format(
            topic=topic,
            prev_section_context=prev_context,
            primary_heading=section.heading,
            primary_content=section.full_content,
            merge_heading=source_section.heading,
            merge_content=source_section.full_content,
            next_section_context=next_context,
            instruction_details=instruction.details,
            target_min=int(combined_words * 0.7),
            target_max=int(combined_words * 1.1),
            combined_word_count=combined_words,
            all_citations=", ".join(all_citations) if all_citations else "none",
        )

        # For validation, we need to preserve citations from both sections
        validation_original = TopLevelSection(
            index=section_index,
            heading=section.heading,
            full_content=section.full_content + "\n" + source_section.full_content,
        )
    else:
        # Standard rewrite/expand/condense
        user_prompt = V2_SECTION_REWRITE_USER.format(
            topic=topic,
            prev_section_context=prev_context,
            section_heading=section.heading,
            section_content=section.full_content,
            next_section_context=next_context,
            instruction_type=instruction.instruction_type,
            instruction_details=instruction.details,
            original_word_count=original_words,
            target_min=target_min,
            target_max=target_max,
            citations=", ".join(section.citations) if section.citations else "none",
        )
        validation_original = section

    # Determine model tier
    use_opus = quality_settings.get("use_opus_for_generation", False)
    tier = ModelTier.OPUS if use_opus else ModelTier.SONNET

    # Call LLM for rewriting
    try:
        llm = get_llm(tier=tier, max_tokens=8000)
        messages = [
            {"role": "system", "content": V2_SECTION_REWRITE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        response = await llm.ainvoke(messages)
        rewritten_content = response.content.strip()
    except Exception as e:
        logger.error(f"Rewrite LLM call failed for section [{section_index}]: {e}")
        return {
            "rewritten_sections": [],
            "errors": [
                {
                    "phase": "rewrite",
                    "section_index": section_index,
                    "error": str(e),
                }
            ],
        }

    # Validate the rewrite
    validation = validate_rewrite(
        validation_original,
        rewritten_content,
        instruction_type=instruction.instruction_type,
        length_tolerance=tolerance,
    )

    if not validation.passes_validation:
        logger.warning(
            f"Section [{section_index}] rewrite failed validation: "
            f"{validation.rejection_reason}"
        )
        # Still return the result, but flag it as failed validation
        # The reassemble phase will decide what to do

    # Log length warnings (soft limits - for review but doesn't fail)
    if validation.length_warning:
        logger.warning(
            f"Section [{section_index}] '{section.heading}': {validation.length_warning}"
        )

    result = RewrittenSection(
        section_index=section_index,
        instruction_type=instruction.instruction_type,
        original_heading=section.heading,
        new_content=rewritten_content,
        validation=validation,
        merge_source_index=instruction.merge_source_index,
    )

    logger.info(
        f"Section [{section_index}] rewritten: "
        f"{validation.original_word_count} -> {validation.rewritten_word_count} words, "
        f"validation={'passed' if validation.passes_validation else 'FAILED'}"
        f"{' (with warning)' if validation.length_warning else ''}"
    )

    # Log full content for expand instructions to monitor synthesis quality
    if instruction.instruction_type == "expand":
        added_words = validation.rewritten_word_count - validation.original_word_count
        logger.debug(
            f"EXPAND content for section [{section_index}] '{section.heading}' "
            f"(+{added_words} words):\n"
            f"{'=' * 60}\n"
            f"{rewritten_content}\n"
            f"{'=' * 60}"
        )

    return {"rewritten_sections": [result.model_dump()]}
