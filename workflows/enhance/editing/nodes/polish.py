"""Polish node for editing workflow."""

import logging
from typing import Any

from langsmith import traceable

from workflows.enhance.editing.document_model import DocumentModel, ContentBlock
from workflows.enhance.editing.schemas import PolishScreeningResult, SectionPolish
from workflows.enhance.editing.prompts import (
    POLISH_SCREENING_SYSTEM,
    POLISH_SCREENING_USER,
    POLISH_SECTION_SYSTEM,
    POLISH_SECTION_USER,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="EditingPolish")
async def polish_node(state: dict) -> dict[str, Any]:
    """Polish document for coherence: transitions, flow improvements.

    Works at section level like other phases:
    1. Screen sections to identify which need polish
    2. Polish each flagged section
    3. Replace section content with polished version
    """
    document_model = DocumentModel.from_dict(
        state.get("updated_document_model", state["document_model"])
    )
    quality_settings = state.get("quality_settings", {})
    max_polish_sections = quality_settings.get("max_polish_sections", 10)

    # Get leaf sections with content
    all_sections = document_model.get_all_sections()
    leaf_sections = [s for s in all_sections if not s.subsections and s.blocks]

    if not leaf_sections:
        logger.info("No sections to polish")
        return {
            "polish_results": [],
            "polish_complete": True,
        }

    logger.info(f"Starting polish phase: screening {len(leaf_sections)} sections")

    # Step 1: Screen sections for polish needs
    sections_to_polish = await _screen_sections_for_polish(
        document_model, leaf_sections, max_polish_sections
    )

    if not sections_to_polish:
        logger.info("No sections need polish work")
        return {
            "polish_results": [],
            "polish_complete": True,
        }

    logger.info(f"Polishing {len(sections_to_polish)} sections")

    # Step 2: Polish each flagged section
    results = []
    for section_id in sections_to_polish:
        section = document_model.get_section(section_id)
        if not section:
            continue

        content = document_model.get_section_content(section_id, include_subsections=False)

        # Skip very short sections
        if len(content.split()) < 50:
            continue

        try:
            polish_result = await get_structured_output(
                output_schema=SectionPolish,
                user_prompt=POLISH_SECTION_USER.format(
                    section_heading=section.heading,
                    section_content=content,
                ),
                system_prompt=POLISH_SECTION_SYSTEM,
                tier=ModelTier.DEEPSEEK_V3,  # DeepSeek V3 for polish (fast/cheap)
                max_tokens=4000,
            )

            # Replace section content with polished version
            if polish_result.polished_content and len(polish_result.polished_content) > 50:
                new_blocks = [
                    ContentBlock.from_content(para, "paragraph")
                    for para in polish_result.polished_content.split("\n\n")
                    if para.strip()
                ]
                section.blocks = new_blocks

                results.append({
                    "section_id": section_id,
                    "section_heading": section.heading,
                    "changes_made": polish_result.changes_made,
                    "success": True,
                })
                logger.debug(
                    f"Polished section '{section.heading}': "
                    f"{len(polish_result.changes_made)} changes"
                )

        except Exception as e:
            logger.warning(f"Polish failed for section '{section.heading}': {e}")
            results.append({
                "section_id": section_id,
                "section_heading": section.heading,
                "success": False,
                "error": str(e),
            })

    successful = sum(1 for r in results if r.get("success"))
    logger.info(f"Polish complete: {successful}/{len(results)} sections polished")

    return {
        "updated_document_model": document_model.to_dict(),
        "polish_results": results,
        "polish_complete": True,
    }


async def _screen_sections_for_polish(
    document_model: DocumentModel,
    leaf_sections: list,
    max_sections: int,
) -> list[str]:
    """Screen sections to identify which need polish work.

    Uses Haiku for fast/cheap screening.
    """
    # Build compact summary for screening
    sections_summary_parts = []
    for section in leaf_sections:
        content = document_model.get_section_content(
            section.section_id, include_subsections=False
        )
        preview = content[:200].replace("\n", " ").strip()
        sections_summary_parts.append(
            f"- {section.section_id}: \"{section.heading}\"\n  Preview: {preview}..."
        )

    sections_summary = "\n".join(sections_summary_parts)

    try:
        result = await get_structured_output(
            output_schema=PolishScreeningResult,
            user_prompt=POLISH_SCREENING_USER.format(
                sections_summary=sections_summary,
            ),
            system_prompt=POLISH_SCREENING_SYSTEM,
            tier=ModelTier.HAIKU,
            max_tokens=2000,
        )

        sections_to_polish = result.sections_to_polish[:max_sections]

        logger.info(
            f"Polish screening: {len(sections_to_polish)} to polish, "
            f"{len(result.sections_ok)} OK. {result.screening_summary}"
        )

        return sections_to_polish

    except Exception as e:
        logger.warning(f"Polish screening failed: {e}")
        # Fallback: polish first N sections
        return [s.section_id for s in leaf_sections[:max_sections]]
