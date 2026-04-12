"""Main entry point for the agent-team web research workflow."""

import logging
import uuid

from langsmith import traceable

from core.config import configure_langsmith
from core.task_queue.task_context import get_trace_metadata, get_trace_tags
from workflows.shared.quality_config import QualityTier

from .config import QUALITY_PRESETS
from .supervisor import run_research

configure_langsmith()
logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="DeepResearchTeam")
async def deep_research_team(
    query: str,
    quality: QualityTier = "standard",
    max_iterations: int | None = None,
    recency_filter: dict | None = None,
) -> dict:
    """Run deep research using an agent team.

    Opus supervisor + parallel Opus researcher subagents.
    Uses claude -p (Max subscription) with MCP tools.

    Args:
        query: Research question or topic
        quality: Quality tier controlling iteration count
        max_iterations: Override iteration count
        recency_filter: Optional {"after_date": "YYYY-MM-DD", "quota": 0.3}

    Returns:
        Same interface as deep_research() for compatibility.
    """
    run_id = uuid.uuid4()
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])
    iterations = max_iterations or preset["max_iterations"]
    max_researcher_turns = preset["max_researcher_turns"]

    recency_info = ""
    if recency_filter:
        recency_info = (
            f"Prioritize sources from after {recency_filter['after_date']}. "
            f"Include year references (2025, 2026) in search queries."
        )

    logger.info(
        f"Starting agent-team research: query='{query[:60]}...', "
        f"quality={quality}, iterations={iterations}, "
        f"researcher_turns={max_researcher_turns}, run_id={run_id}"
    )

    result = await run_research(
        query=query,
        max_iterations=iterations,
        max_researcher_turns=max_researcher_turns,
        recency_info=recency_info,
    )

    return {
        "final_report": result["final_report"],
        "status": result["status"],
        "langsmith_run_id": str(run_id),
        "errors": result["errors"],
        "source_count": result["source_count"],
        "citation_keys": [],
        "started_at": result["started_at"],
        "completed_at": result["completed_at"],
    }
