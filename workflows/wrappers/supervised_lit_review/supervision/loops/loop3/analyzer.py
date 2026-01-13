"""Phase A/B analysis for structural editing.

Phase A: Identify structural issues (diagnosis)
Phase B: Generate concrete edits based on identified issues (prescription)
"""

import json
import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.wrappers.supervised_lit_review.supervision.types import (
    EditManifest,
    StructuralIssueAnalysis,
)
from workflows.wrappers.supervised_lit_review.supervision.prompts import (
    LOOP3_PHASE_A_SYSTEM,
    LOOP3_PHASE_A_USER,
    LOOP3_PHASE_B_SYSTEM,
    LOOP3_PHASE_B_USER,
)

logger = logging.getLogger(__name__)


async def analyze_structure_phase_a_node(state: dict) -> dict[str, Any]:
    """Phase A: Identify structural issues without generating edits.

    Uses Opus with extended thinking to carefully assess structural issues.
    Returns issue analysis that feeds into Phase B for edit generation.

    This separation gives the LLM a clear cognitive path:
    1. Phase A: Diagnose problems
    2. Phase B: Prescribe fixes
    """
    numbered_doc = state["numbered_document"]
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    topic = state.get("topic", "")

    user_prompt = LOOP3_PHASE_A_USER.format(
        numbered_document=numbered_doc,
        topic=topic,
        iteration=iteration + 1,
        max_iterations=max_iterations,
    )

    try:
        analysis = await get_structured_output(
            output_schema=StructuralIssueAnalysis,
            user_prompt=user_prompt,
            system_prompt=LOOP3_PHASE_A_SYSTEM,
            tier=ModelTier.OPUS,
            thinking_budget=6000,
            max_tokens=8000,
            use_json_schema_method=True,
            max_retries=2,
        )

        logger.info(
            f"Loop 3 Phase A complete: needs_restructuring={analysis.needs_restructuring}, "
            f"issues_found={len(analysis.issues)}"
        )

        for issue in analysis.issues:
            num_paragraphs = len(issue.affected_paragraphs)
            log_fn = logger.warning if num_paragraphs > 25 else logger.debug
            log_fn(
                f"Issue {issue.issue_id} ({issue.issue_type}): "
                f"{num_paragraphs} paragraphs affected - {issue.description}"
            )

        return {
            "issue_analysis": analysis.model_dump(),
            "phase_a_complete": True,
        }

    except Exception as e:
        logger.error(f"Phase A analysis failed: {e}", exc_info=True)
        return {
            "issue_analysis": {
                "issues": [],
                "overall_assessment": f"Phase A failed: {e}",
                "needs_restructuring": False,
            },
            "phase_a_complete": True,
        }


def validate_issue_edit_mapping(
    issues: list[dict],
    edits: list[dict],
) -> tuple[list[int], list[int]]:
    """Validate that Phase A issues have corresponding edits from Phase B.

    Extracts issue_ids referenced in edit notes and compares to Phase A issues.
    This is informational validation - logs warnings but doesn't block execution.

    Args:
        issues: List of issues from Phase A (each with issue_id)
        edits: List of edits from Phase B (issue_id may be in notes)

    Returns:
        Tuple of (mapped_issue_ids, unmapped_issue_ids)
    """
    import re

    phase_a_ids = {issue.get("issue_id") for issue in issues if issue.get("issue_id") is not None}

    mapped_ids = set()
    for edit in edits:
        notes = edit.get("notes", "")
        matches = re.findall(r'issue[_\s]?(?:id)?[:\s]*(\d+)', notes.lower())
        mapped_ids.update(int(m) for m in matches)

    unmapped = phase_a_ids - mapped_ids
    return list(mapped_ids), list(unmapped)


async def generate_edits_phase_b_node(state: dict) -> dict[str, Any]:
    """Phase B: Generate concrete edits based on identified issues.

    Takes the issues from Phase A and generates specific edits.
    Each issue should map to one or more edits in the manifest.

    Note: This is legacy code kept for backward compatibility.
    The current graph uses rewrite_sections_for_issues_node instead.
    """
    numbered_doc = state["numbered_document"]
    issue_analysis = state.get("issue_analysis", {})
    topic = state.get("topic", "")

    issues = issue_analysis.get("issues", [])

    if not issues:
        logger.debug("Phase B called with no issues")
        return {
            "edit_manifest": {
                "edits": [],
                "todo_markers": [],
                "overall_assessment": "No issues to resolve",
                "needs_restructuring": False,
            }
        }

    issues_json = json.dumps(issues, indent=2)

    user_prompt = LOOP3_PHASE_B_USER.format(
        issues_json=issues_json,
        numbered_document=numbered_doc,
        topic=topic,
    )

    try:
        manifest = await get_structured_output(
            output_schema=EditManifest,
            user_prompt=user_prompt,
            system_prompt=LOOP3_PHASE_B_SYSTEM,
            tier=ModelTier.OPUS,
            thinking_budget=6000,
            max_tokens=10000,
            use_json_schema_method=True,
            max_retries=2,
        )

        manifest_dict = manifest.model_dump()
        if not manifest_dict.get("architecture_assessment") and issue_analysis.get("architecture_assessment"):
            manifest_dict["architecture_assessment"] = issue_analysis["architecture_assessment"]

        logger.info(
            f"Loop 3 Phase B complete: generated {len(manifest.edits)} edits, "
            f"{len(manifest.todo_markers)} todos"
        )

        if manifest.edits and issues:
            edit_dicts = [e.model_dump() for e in manifest.edits]
            mapped_ids, unmapped_ids = validate_issue_edit_mapping(issues, edit_dicts)
            if unmapped_ids:
                unmapped_pct = len(unmapped_ids) / len(issues) * 100
                logger.warning(
                    f"Phase B: {len(unmapped_ids)}/{len(issues)} issues unmapped "
                    f"({unmapped_pct:.0f}%): issue_ids {unmapped_ids}"
                )

        return {"edit_manifest": manifest_dict}

    except Exception as e:
        logger.error(f"Phase B edit generation failed: {e}", exc_info=True)
        todos = [
            f"Resolve issue {i['issue_id']}: {i['description'][:100]}"
            for i in issues[:5]
        ]
        return {
            "edit_manifest": {
                "edits": [],
                "todo_markers": todos,
                "overall_assessment": f"Edit generation failed: {e}. Manual review needed.",
                "needs_restructuring": False,
                "architecture_assessment": issue_analysis.get("architecture_assessment"),
            }
        }
