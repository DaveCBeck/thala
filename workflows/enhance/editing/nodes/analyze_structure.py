"""Analyze structure node for editing workflow."""

import logging
from typing import Any

from langsmith import traceable

from workflows.enhance.editing.document_model import DocumentModel
from workflows.enhance.editing.schemas import StructuralAnalysis
from workflows.enhance.editing.prompts import (
    STRUCTURE_ANALYSIS_SYSTEM,
    STRUCTURE_ANALYSIS_USER,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="EditingAnalyzeStructure")
async def analyze_structure_node(state: dict) -> dict[str, Any]:
    """Analyze document structure to identify issues.

    Uses the full document (suitable for caching) to identify
    structural problems that need to be fixed.

    Args:
        state: Current workflow state

    Returns:
        State update with structural_analysis
    """
    # Use updated document model if available (from previous iterations)
    document_model = DocumentModel.from_dict(
        state.get("updated_document_model", state["document_model"])
    )
    topic = state["input"]["topic"]
    iteration = state.get("structure_iteration", 0)
    max_iterations = state.get("max_structure_iterations", 3)

    # Get quality settings
    quality_settings = state.get("quality_settings", {})
    use_opus = quality_settings.get("use_opus_for_analysis", True)
    thinking_budget = quality_settings.get("analysis_thinking_budget", 6000)

    # Render document for analysis (with stable IDs)
    document_xml = document_model.render_for_analysis()

    # Build focus instruction based on iteration
    if iteration == 0:
        focus_instruction = "This is the first analysis pass. Identify all structural issues."
    else:
        focus_instruction = (
            f"This is iteration {iteration + 1}. Focus on issues remaining "
            "after previous fixes. Check if earlier fixes were effective."
        )

    user_prompt = STRUCTURE_ANALYSIS_USER.format(
        topic=topic,
        iteration=iteration + 1,
        max_iterations=max_iterations,
        focus_instruction=focus_instruction,
        document_xml=document_xml,
    )

    logger.info(
        f"Analyzing structure (iteration {iteration + 1}/{max_iterations}), "
        f"doc: {document_model.total_words} words"
    )

    try:
        analysis = await get_structured_output(
            output_schema=StructuralAnalysis,
            user_prompt=user_prompt,
            system_prompt=STRUCTURE_ANALYSIS_SYSTEM,
            tier=ModelTier.OPUS if use_opus else ModelTier.SONNET,
            thinking_budget=thinking_budget if use_opus else None,
            max_tokens=8000,
            use_json_schema_method=True,
            max_retries=2,
        )

        logger.info(
            f"Analysis complete: coherence={analysis.narrative_coherence_score:.2f}, "
            f"organization={analysis.section_organization_score:.2f}, "
            f"issues={len(analysis.issues)} "
            f"(critical={analysis.critical_issues_count}, major={analysis.major_issues_count})"
        )

        # Log issues at appropriate levels
        for issue in analysis.issues:
            log_fn = logger.warning if issue.severity in ("critical", "major") else logger.debug
            log_fn(
                f"Issue {issue.issue_id} ({issue.issue_type}, {issue.severity}): "
                f"{issue.description[:100]}"
            )

        return {
            "structural_analysis": analysis.model_dump(),
            "analysis_complete": True,
        }

    except Exception as e:
        logger.error(f"Structure analysis failed: {e}", exc_info=True)
        # Return a pass-through analysis on failure
        return {
            "structural_analysis": StructuralAnalysis(
                has_clear_introduction=True,
                has_clear_conclusion=True,
                narrative_coherence_score=0.7,
                section_organization_score=0.7,
                issues=[],
                overall_assessment=f"Analysis failed: {e}. Proceeding without structural changes.",
            ).model_dump(),
            "analysis_complete": True,
            "errors": [{"node": "analyze_structure", "error": str(e)}],
        }
