"""Aggregation functions for deep research graph."""

import logging
from typing import Any

from workflows.research.web_research.state import (
    DeepResearchState,
    calculate_completeness,
)

logger = logging.getLogger(__name__)


def aggregate_researcher_findings(state: DeepResearchState) -> dict[str, Any]:
    """Aggregate findings from researcher agents back to main state.

    This is called after researcher agents complete. The findings are
    automatically aggregated via the Annotated[..., add] pattern.
    Also updates completeness based on accumulated findings.
    """
    findings = state.get("research_findings", [])
    diffusion = state.get("diffusion", {})
    brief = state.get("research_brief", {})
    draft = state.get("draft_report")

    # Calculate updated completeness based on new findings
    new_completeness = calculate_completeness(
        findings=findings,
        key_questions=brief.get("key_questions", []),
        iteration=diffusion.get("iteration", 0),
        max_iterations=diffusion.get("max_iterations", 4),
        gaps_remaining=draft.get("gaps_remaining", []) if draft else [],
    )

    logger.debug(
        f"Aggregated {len(findings)} research findings, completeness: {new_completeness:.0%}"
    )

    return {
        "pending_questions": [],
        "current_status": "supervising",
        "diffusion": {
            **diffusion,
            "completeness_score": new_completeness,
        },
    }
