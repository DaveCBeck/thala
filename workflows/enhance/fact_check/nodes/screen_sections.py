"""Screen sections node for fact-check workflow."""

import logging
from typing import Any

from langsmith import traceable

from workflows.enhance.editing.document_model import DocumentModel
from workflows.enhance.fact_check.schemas import FactCheckScreeningResult
from workflows.enhance.fact_check.prompts import (
    FACT_CHECK_SCREENING_SYSTEM,
    FACT_CHECK_SCREENING_USER,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="FactCheckScreenSections")
async def screen_sections_for_fact_check(state: dict) -> dict[str, Any]:
    """Pre-screen sections to determine which need fact-checking.

    Uses a lightweight LLM call to categorize sections and identify which
    contain verifiable claims worth checking.

    Returns:
        State update with screened_sections list containing section IDs to check.
    """
    document_model_dict = state.get("updated_document_model", state.get("document_model"))
    if not document_model_dict:
        return {"screened_sections": [], "screening_skipped": []}

    document_model = DocumentModel.from_dict(document_model_dict)
    topic = state["input"]["topic"]

    # Get leaf sections with content
    all_sections = document_model.get_all_sections()
    leaf_sections = [s for s in all_sections if not s.subsections and s.blocks]

    if not leaf_sections:
        return {"screened_sections": [], "screening_skipped": []}

    # Build compact summary for screening (heading + first 150 chars)
    sections_summary_parts = []
    for section in leaf_sections:
        content = document_model.get_section_content(section.section_id, include_subsections=False)
        preview = content[:150].replace("\n", " ").strip()
        sections_summary_parts.append(
            f"- {section.section_id}: \"{section.heading}\"\n  Preview: {preview}..."
        )

    sections_summary = "\n".join(sections_summary_parts)

    logger.debug(f"Pre-screening {len(leaf_sections)} sections for fact-check priority")

    try:
        result = await get_structured_output(
            output_schema=FactCheckScreeningResult,
            user_prompt=FACT_CHECK_SCREENING_USER.format(
                topic=topic,
                sections_summary=sections_summary,
            ),
            system_prompt=FACT_CHECK_SCREENING_SYSTEM,
            tier=ModelTier.DEEPSEEK_V3,  # DeepSeek V3 for screening
            max_tokens=2000,  # Simplified schema needs less tokens
        )

        sections_to_check = result.sections_to_check
        sections_to_skip = result.sections_to_skip

        logger.info(
            f"Fact-check screening: {len(sections_to_check)} to check, "
            f"{len(sections_to_skip)} to skip. {result.screening_summary}"
        )

        return {
            "screened_sections": sections_to_check,
            "screening_skipped": sections_to_skip,
        }

    except Exception as e:
        logger.warning(f"Fact-check screening failed, checking all sections: {e}")
        # Fallback: check all sections
        return {
            "screened_sections": [s.section_id for s in leaf_sections],
            "screening_skipped": [],
        }
