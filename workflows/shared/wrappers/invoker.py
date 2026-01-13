"""
Standardized workflow invocation.

Provides a single entry point for invoking any registered workflow
with normalized parameters and result handling.
"""

import logging
from typing import Any

from workflows.shared.wrappers.registry import get_workflow, get_available_workflows
from workflows.shared.wrappers.quality import get_workflow_quality, QUALITY_MAPPING
from workflows.shared.wrappers.result_types import WorkflowResult

logger = logging.getLogger(__name__)


async def invoke_workflow(
    workflow_key: str,
    *,
    query: str,
    quality: str = "standard",
    language: str = "en",
    research_questions: list[str] | None = None,
    date_range: tuple[int, int] | None = None,
    **kwargs: Any,
) -> WorkflowResult:
    """Invoke a registered workflow with normalized parameters.

    This function handles parameter mapping and provides a consistent
    interface for invoking any registered workflow.

    Args:
        workflow_key: The workflow to invoke (e.g., "web_research", "academic_lit_review")
        query: The research query/topic/theme
        quality: Unified quality tier ("quick", "standard", "comprehensive")
        language: ISO 639-1 language code (default: "en")
        research_questions: Optional research questions (for workflows that require them)
        date_range: Optional (start_year, end_year) tuple
        **kwargs: Additional workflow-specific parameters

    Returns:
        WorkflowResult with final_report, source_count, status, and errors

    Raises:
        ValueError: If workflow_key is not registered
        KeyError: If quality tier is not valid
    """
    config = get_workflow(workflow_key)
    if not config:
        available = get_available_workflows()
        raise ValueError(
            f"Unknown workflow: {workflow_key}. "
            f"Available workflows: {', '.join(available)}"
        )

    # Validate quality tier
    if quality not in QUALITY_MAPPING:
        raise KeyError(
            f"Unknown quality tier: {quality}. "
            f"Valid tiers: {', '.join(QUALITY_MAPPING.keys())}"
        )

    # Get workflow-specific quality setting
    workflow_quality = get_workflow_quality(quality, workflow_key)

    # Build invocation arguments
    invoke_kwargs: dict[str, Any] = {
        "query": query,
        "quality": workflow_quality,
        "language": language,
    }

    # Add research_questions if workflow requires them
    if config["requires_questions"]:
        if not research_questions:
            # Generate default questions
            research_questions = [
                f"What is the current state of research on {query}?",
                f"What are the key findings and debates about {query}?",
            ]
        invoke_kwargs["research_questions"] = research_questions

    # Add date_range if workflow supports it and it's provided
    if config["supports_date_range"] and date_range:
        invoke_kwargs["date_range"] = date_range

    # Add any additional kwargs
    invoke_kwargs.update(kwargs)

    logger.debug(
        f"Invoking workflow: {workflow_key} "
        f"(quality: {quality} -> {workflow_quality}, language: {language})"
    )

    try:
        result = await config["runner"](**invoke_kwargs)
        return result
    except Exception as e:
        logger.error(f"Workflow {workflow_key} failed: {e}")
        return WorkflowResult(
            final_report=None,
            source_count=0,
            status="failed",
            errors=[{"phase": workflow_key, "error": str(e)}],
        )
