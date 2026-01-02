"""
Parallel research node that runs web and academic research simultaneously.

Uses asyncio.gather to run both workflows in parallel, waiting for both
to complete before proceeding.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from workflows.research.graph.api import deep_research
from workflows.research.subgraphs.academic_lit_review import academic_lit_review
from workflows.wrapped.state import WrappedResearchState, WorkflowResult, QUALITY_MAPPING

logger = logging.getLogger(__name__)


async def run_parallel_research(state: WrappedResearchState) -> dict[str, Any]:
    """Run web and academic research workflows simultaneously.

    Uses asyncio.gather to run both in parallel, waits for both to complete.
    Each workflow is wrapped to capture errors independently so one failure
    doesn't prevent the other from completing.
    """
    input_data = state["input"]
    quality = input_data["quality"]
    quality_config = QUALITY_MAPPING[quality]

    async def run_web() -> WorkflowResult:
        """Run web research with error handling."""
        started_at = datetime.utcnow()
        try:
            logger.info(f"Starting web research with depth={quality_config['web_depth']}")
            result = await deep_research(
                query=input_data["query"],
                depth=quality_config["web_depth"],
            )
            return WorkflowResult(
                workflow_type="web",
                final_output=result.get("final_report"),
                started_at=started_at,
                completed_at=datetime.utcnow(),
                status="completed",
                error=None,
                top_of_mind_id=None,
            )
        except Exception as e:
            logger.error(f"Web research failed: {e}")
            return WorkflowResult(
                workflow_type="web",
                final_output=None,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                status="failed",
                error=str(e),
                top_of_mind_id=None,
            )

    async def run_academic() -> WorkflowResult:
        """Run academic lit review with error handling."""
        started_at = datetime.utcnow()

        # Generate research questions if not provided
        research_questions = input_data.get("research_questions") or [
            f"What are the main research themes in {input_data['query']}?",
            f"What methodological approaches are used to study {input_data['query']}?",
            f"What are the key findings and debates in this area?",
        ]

        try:
            logger.info(f"Starting academic research with quality={quality_config['academic_quality']}")
            result = await academic_lit_review(
                topic=input_data["query"],
                research_questions=research_questions,
                quality=quality_config["academic_quality"],
                date_range=input_data.get("date_range"),
            )
            return WorkflowResult(
                workflow_type="academic",
                final_output=result.get("final_review"),
                started_at=started_at,
                completed_at=datetime.utcnow(),
                status="completed",
                error=None,
                top_of_mind_id=None,
            )
        except Exception as e:
            logger.error(f"Academic research failed: {e}")
            return WorkflowResult(
                workflow_type="academic",
                final_output=None,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                status="failed",
                error=str(e),
                top_of_mind_id=None,
            )

    # Run both in parallel
    logger.info(
        f"Starting parallel research for query: {input_data['query'][:50]}... "
        f"(web: {quality_config['web_depth']}, academic: {quality_config['academic_quality']})"
    )
    web_result, academic_result = await asyncio.gather(run_web(), run_academic())

    # Build return state update
    updates: dict[str, Any] = {
        "web_result": web_result,
        "academic_result": academic_result,
        "current_phase": "parallel_research_complete",
    }

    # Collect errors if any
    errors = []
    if web_result["status"] == "failed":
        errors.append({"phase": "web_research", "error": web_result["error"]})
    if academic_result["status"] == "failed":
        errors.append({"phase": "academic_research", "error": academic_result["error"]})

    if errors:
        updates["errors"] = errors

    logger.info(
        f"Parallel research complete. Web: {web_result['status']}, Academic: {academic_result['status']}"
    )

    return updates
