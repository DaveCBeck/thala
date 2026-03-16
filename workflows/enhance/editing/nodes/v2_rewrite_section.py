"""V2 Phase 2: Section rewriting node.

This node rewrites a single section following the edit instruction.
It runs in parallel for each section that needs work.
"""

import logging
from typing import Any

from langsmith import traceable

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier

from ..prompts import V2_SECTION_REWRITE_SYSTEM, V2_SECTION_REWRITE_USER, V2_SECTION_MERGE_USER
from ..schemas import (
    EditInstruction,
    RewrittenSection,
    SectionValidation,
    TopLevelSection,
    extract_v2_citations,
)

logger = logging.getLogger(__name__)


def get_context_window(sections: list[TopLevelSection], index: int, position: str, max_words: int = 500) -> str:
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
    target_min: int,
    target_max: int,
) -> SectionValidation:
    """Validate a rewritten section against target word-count bounds.

    Fails validation when the output deviates drastically from the
    targets given to the LLM (below 50% of target_min or above 150%
    of target_max). Smaller deviations are acceptable — the targets
    are advisory and some drift is normal.

    Also checks that all original citations are preserved.

    Args:
        original: Original section
        rewritten_content: Rewritten content
        target_min: Minimum word count given to the LLM
        target_max: Maximum word count given to the LLM

    Returns:
        SectionValidation with results
    """
    rewritten_words = len(rewritten_content.split())

    original_citations = original.citations
    rewritten_citations = extract_v2_citations(rewritten_content)

    # Hard bounds: 50% of target_min floor, 150% of target_max ceiling
    hard_floor = int(target_min * 0.5)
    hard_ceiling = int(target_max * 1.5)

    length_ok = hard_floor <= rewritten_words <= hard_ceiling

    passes = length_ok
    rejection_reason = None
    length_warning = None

    if not length_ok:
        if rewritten_words < hard_floor:
            rejection_reason = (
                f"Extreme reduction: {rewritten_words} words "
                f"(target_min was {target_min}, hard floor {hard_floor})"
            )
        else:
            rejection_reason = (
                f"Extreme expansion: {rewritten_words} words "
                f"(target_max was {target_max}, hard ceiling {hard_ceiling})"
            )

    # Log dropped citations as informational — editorial rewrites may
    # legitimately remove citations that no longer fit the narrative.
    missing_cites = set(original_citations) - set(rewritten_citations)
    if missing_cites:
        logger.info(f"Citations dropped during rewrite: {', '.join(missing_cites)}")

    # Advisory warning for outputs outside the target range but within hard bounds
    if length_ok and rewritten_words < target_min:
        length_warning = (
            f"Below target: {rewritten_words} words (target_min {target_min})"
        )
    elif length_ok and rewritten_words > target_max:
        length_warning = (
            f"Above target: {rewritten_words} words (target_max {target_max})"
        )

    return SectionValidation(
        original_word_count=original.word_count,
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

    logger.info(f"Rewriting section [{section_index}] '{section.heading}' ({instruction.instruction_type})")

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

        target_min = int(combined_words * 0.7)
        target_max = int(combined_words * 1.1)

        user_prompt = V2_SECTION_MERGE_USER.format(
            topic=topic,
            prev_section_context=prev_context,
            primary_heading=section.heading,
            primary_content=section.full_content,
            merge_heading=source_section.heading,
            merge_content=source_section.full_content,
            next_section_context=next_context,
            instruction_details=instruction.details,
            target_min=target_min,
            target_max=target_max,
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

    # Attempt rewrite with one retry on validation failure
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            response = await invoke(
                tier=tier,
                system=V2_SECTION_REWRITE_SYSTEM,
                user=user_prompt,
                config=InvokeConfig(
                    max_tokens=64000,
                    batch_policy=BatchPolicy.PREFER_BALANCE,
                ),
            )
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

        validation = validate_rewrite(
            validation_original,
            rewritten_content,
            target_min=target_min,
            target_max=target_max,
        )

        logger.info(
            f"Section [{section_index}] rewrite attempt {attempt}: "
            f"{validation.original_word_count} -> {validation.rewritten_word_count} words, "
            f"validation={'passed' if validation.passes_validation else 'FAILED'}"
            f"{f' ({validation.length_warning})' if validation.length_warning else ''}"
        )

        if validation.passes_validation:
            break

        logger.warning(
            f"Section [{section_index}] rewrite failed validation: "
            f"{validation.rejection_reason}"
        )

        if attempt < max_attempts:
            logger.info(f"Section [{section_index}] retrying rewrite (attempt {attempt + 1})")

    # If validation still fails after retry, fall back to original section
    if not validation.passes_validation:
        logger.warning(
            f"Section [{section_index}] '{section.heading}': rewrite failed validation "
            f"after {max_attempts} attempts, keeping original section"
        )
        result = RewrittenSection(
            section_index=section_index,
            instruction_type=instruction.instruction_type,
            original_heading=section.heading,
            new_content=section.full_content,
            validation=SectionValidation(
                original_word_count=section.word_count,
                rewritten_word_count=section.word_count,
                original_citations=section.citations,
                rewritten_citations=section.citations,
                passes_validation=True,
                rejection_reason=None,
                length_warning=f"Fell back to original after failed rewrite: {validation.rejection_reason}",
            ),
            merge_source_index=instruction.merge_source_index,
        )
        return {"rewritten_sections": [result.model_dump()]}

    if validation.length_warning:
        logger.info(f"Section [{section_index}] '{section.heading}': {validation.length_warning}")

    result = RewrittenSection(
        section_index=section_index,
        instruction_type=instruction.instruction_type,
        original_heading=section.heading,
        new_content=rewritten_content,
        validation=validation,
        merge_source_index=instruction.merge_source_index,
    )

    return {"rewritten_sections": [result.model_dump()]}
