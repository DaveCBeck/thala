"""Enhance section worker for editing workflow."""

import logging
from typing import Any

from langgraph.types import Send

from workflows.enhance.editing.document_model import DocumentModel, Section
from workflows.enhance.editing.schemas import SectionEnhancement
from workflows.enhance.editing.prompts import (
    ENHANCE_ABSTRACT_SYSTEM,
    ENHANCE_ABSTRACT_USER,
    ENHANCE_FRAMING_SYSTEM,
    ENHANCE_FRAMING_USER,
    ENHANCE_CONTENT_SYSTEM,
    ENHANCE_CONTENT_USER,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


def get_section_type(section: Section) -> str:
    """Determine section type for prompt selection."""
    heading_lower = section.heading.lower()

    if "abstract" in heading_lower:
        return "abstract"
    elif any(kw in heading_lower for kw in ["introduction", "overview", "background"]):
        return "introduction"
    elif any(kw in heading_lower for kw in ["conclusion", "summary", "future"]):
        return "conclusion"
    elif "methodology" in heading_lower or "methods" in heading_lower:
        return "methodology"
    else:
        return "content"


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def check_word_count_tolerance(
    original_count: int,
    enhanced_count: int,
    tolerance: float,
) -> tuple[bool, float]:
    """Check if enhanced word count is within tolerance.

    Args:
        original_count: Original word count
        enhanced_count: Enhanced word count
        tolerance: Tolerance as fraction (e.g., 0.20 for ±20%)

    Returns:
        Tuple of (is_within_tolerance, change_percent)
    """
    if original_count == 0:
        return True, 0.0

    change = enhanced_count - original_count
    change_percent = change / original_count

    return abs(change_percent) <= tolerance, change_percent


def route_to_enhance_sections(state: dict) -> list[Send] | str:
    """Route to section enhancement workers.

    Returns list of Send objects for parallel section enhancement,
    or "enhance_coherence_review" if no sections to enhance.
    """
    document_model_dict = state.get("updated_document_model", state.get("document_model"))
    if not document_model_dict:
        return "enhance_coherence_review"

    document_model = DocumentModel.from_dict(document_model_dict)
    iteration = state.get("enhance_iteration", 0)
    flagged_sections = state.get("enhance_flagged_sections", [])

    # Get all sections
    all_sections = document_model.get_all_sections()

    # Filter to leaf sections only (no subsections) to avoid duplicating content
    # Parent sections would include all child content, causing massive word count issues
    leaf_sections = [s for s in all_sections if not s.subsections]

    # On first iteration, enhance all content sections
    # On subsequent iterations, only enhance flagged sections
    if iteration == 0:
        sections_to_enhance = [
            s for s in leaf_sections
            if get_section_type(s) not in ("abstract",)  # Skip abstracts on first pass
        ]
    else:
        sections_to_enhance = [
            s for s in leaf_sections
            if s.section_id in flagged_sections
        ]

    if not sections_to_enhance:
        return "enhance_coherence_review"

    # Build Send objects for parallel enhancement
    sends = []
    for section in sections_to_enhance:
        # Get only this section's content, not subsections
        section_content = document_model.get_section_content(
            section.section_id, include_subsections=False
        )

        # Skip sections with very little content (just headings)
        if count_words(section_content) < 50:
            continue

        sends.append(
            Send(
                "enhance_section",
                {
                    "section_id": section.section_id,
                    "section_content": section_content,
                    "section_heading": section.heading,
                    "section_type": get_section_type(section),
                    "topic": state["input"]["topic"],
                    "quality_settings": state.get("quality_settings", {}),
                    "document_model": document_model_dict,
                },
            )
        )

    if not sends:
        return "enhance_coherence_review"

    logger.info(
        f"Routing to enhance {len(sends)} leaf sections (iteration {iteration}, "
        f"skipped {len(leaf_sections) - len(sends)} small/abstract sections)"
    )
    return sends


async def enhance_section_worker(state: dict) -> dict[str, Any]:
    """Enhance a single section using paper corpus tools.

    This worker:
    1. Selects type-appropriate prompt (abstract, framing, content)
    2. Uses paper search and content tools to find supporting evidence
    3. Enhances arguments while maintaining word count tolerance
    4. Returns the enhanced content with metadata
    """
    section_id = state["section_id"]
    section_content = state["section_content"]
    section_heading = state["section_heading"]
    section_type = state["section_type"]
    topic = state["topic"]
    quality_settings = state.get("quality_settings", {})

    original_word_count = count_words(section_content)
    tolerance = quality_settings.get("enhance_word_tolerance", 0.20)
    use_opus = quality_settings.get("use_opus_for_generation", False)
    max_tool_calls = quality_settings.get("enhance_max_tool_calls", 10)

    logger.info(
        f"Enhancing section '{section_heading}' ({section_type}): "
        f"{original_word_count} words, max_tools={max_tool_calls}"
    )

    # Get paper tools
    from langchain_tools import search_papers, get_paper_content
    tools = [search_papers, get_paper_content]

    # Select prompts based on section type
    if section_type == "abstract":
        system_prompt = ENHANCE_ABSTRACT_SYSTEM
        user_prompt = ENHANCE_ABSTRACT_USER.format(
            section_heading=section_heading,
            section_content=section_content,
            topic=topic,
        )
        max_tokens = 2000
    elif section_type in ("introduction", "conclusion"):
        system_prompt = ENHANCE_FRAMING_SYSTEM
        user_prompt = ENHANCE_FRAMING_USER.format(
            section_type=section_type,
            section_heading=section_heading,
            section_content=section_content,
            topic=topic,
            original_word_count=original_word_count,
            tolerance_percent=int(tolerance * 100),
        )
        max_tokens = 4000
    else:
        system_prompt = ENHANCE_CONTENT_SYSTEM
        user_prompt = ENHANCE_CONTENT_USER.format(
            section_heading=section_heading,
            section_content=section_content,
            topic=topic,
            original_word_count=original_word_count,
            tolerance_percent=int(tolerance * 100),
        )
        max_tokens = 8000

    try:
        result = await get_structured_output(
            output_schema=SectionEnhancement,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tier=ModelTier.OPUS if use_opus else ModelTier.SONNET,
            tools=tools,
            max_tokens=max_tokens,
            max_tool_calls=max_tool_calls,
            use_json_schema_method=True,
        )

        # Validate word count
        enhanced_word_count = count_words(result.enhanced_content)
        within_tolerance, change_percent = check_word_count_tolerance(
            original_word_count, enhanced_word_count, tolerance
        )

        if not within_tolerance and section_type not in ("abstract",):
            logger.warning(
                f"Section '{section_heading}' word count change {change_percent:.1%} "
                f"exceeds tolerance ±{tolerance:.0%}. Keeping enhancement anyway."
            )

        logger.info(
            f"Enhanced section '{section_heading}': "
            f"{original_word_count} → {enhanced_word_count} words "
            f"({change_percent:+.1%}), confidence={result.confidence:.2f}"
        )

        return {
            "section_enhancements": [
                {
                    "section_id": section_id,
                    "section_heading": section_heading,
                    "original_content": section_content,
                    "enhanced_content": result.enhanced_content,
                    "original_word_count": original_word_count,
                    "enhanced_word_count": enhanced_word_count,
                    "citations_added": result.citations_added,
                    "citations_removed": result.citations_removed,
                    "enhancement_notes": result.enhancement_notes,
                    "confidence": result.confidence,
                    "success": True,
                }
            ]
        }

    except Exception as e:
        logger.error(f"Enhancement failed for section '{section_heading}': {e}")
        return {
            "section_enhancements": [
                {
                    "section_id": section_id,
                    "section_heading": section_heading,
                    "success": False,
                    "error": str(e),
                }
            ],
            "errors": [{"node": "enhance_section", "error": str(e)}],
        }


async def assemble_enhancements_node(state: dict) -> dict[str, Any]:
    """Assemble section enhancements into updated document model.

    Applies successful enhancements to the document model.
    """
    document_model = DocumentModel.from_dict(
        state.get("updated_document_model", state["document_model"])
    )
    enhancements = state.get("section_enhancements", [])

    successful = [e for e in enhancements if e.get("success")]
    failed = [e for e in enhancements if not e.get("success")]

    if failed:
        logger.warning(f"{len(failed)} section enhancements failed")

    # Apply successful enhancements
    for enhancement in successful:
        section_id = enhancement["section_id"]
        enhanced_content = enhancement["enhanced_content"]

        section = document_model.get_section(section_id)
        if section:
            # Replace section content with enhanced version
            # This is a simplified approach - we replace all blocks
            from workflows.enhance.editing.document_model import ContentBlock
            new_blocks = [
                ContentBlock.from_content(para, "paragraph")
                for para in enhanced_content.split("\n\n")
                if para.strip()
            ]
            section.blocks = new_blocks

            logger.debug(f"Applied enhancement to section '{section.heading}'")

    logger.info(f"Assembled {len(successful)} section enhancements")

    return {
        "updated_document_model": document_model.to_dict(),
    }
