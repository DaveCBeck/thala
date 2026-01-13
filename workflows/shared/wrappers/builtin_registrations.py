"""
Built-in workflow registrations.

This module registers the four core workflows:
- web_research: Deep web research using search and content scraping
- academic_lit_review: Academic paper discovery, processing, and synthesis
- book_finding: Book discovery and recommendation synthesis
- supervised_lit_review: Academic lit review with multi-loop supervision

These are registered automatically when this module is imported.
Import this module to ensure all built-in workflows are available.
"""

import logging
from workflows.shared.wrappers.registry import register_workflow
from workflows.shared.wrappers.result_types import WorkflowResult

logger = logging.getLogger(__name__)


async def _run_web_research(
    query: str,
    quality: str,
    language: str = "en",
    **kwargs,
) -> WorkflowResult:
    """Adapter for web_research workflow."""
    from workflows.research.web_research.graph.api import deep_research

    result = await deep_research(
        query=query,
        quality=quality,
        language=language,
    )

    return WorkflowResult(
        final_report=result.get("final_report"),
        source_count=result.get("source_count", 0),
        status=result.get("status", "success" if result.get("final_report") else "failed"),
        errors=result.get("errors", []),
    )


async def _run_academic_lit_review(
    query: str,
    quality: str,
    language: str = "en",
    research_questions: list[str] | None = None,
    date_range: tuple[int, int] | None = None,
    **kwargs,
) -> WorkflowResult:
    """Adapter for academic_lit_review workflow."""
    from workflows.research.academic_lit_review.graph.api import academic_lit_review

    result = await academic_lit_review(
        topic=query,
        research_questions=research_questions or [],
        quality=quality,
        language=language,
        date_range=date_range,
    )

    return WorkflowResult(
        final_report=result.get("final_report"),
        source_count=result.get("source_count", 0),
        status=result.get("status", "success" if result.get("final_report") else "failed"),
        errors=result.get("errors", []),
    )


async def _run_book_finding(
    query: str,
    quality: str,
    language: str = "en",
    **kwargs,
) -> WorkflowResult:
    """Adapter for book_finding workflow."""
    from workflows.research.book_finding.graph.api import book_finding

    result = await book_finding(
        theme=query,
        quality=quality,
        language=language,
    )

    return WorkflowResult(
        final_report=result.get("final_report"),
        source_count=result.get("source_count", 0),
        status=result.get("status", "success" if result.get("final_report") else "failed"),
        errors=result.get("errors", []),
    )


async def _run_supervised_lit_review(
    query: str,
    quality: str,
    language: str = "en",
    research_questions: list[str] | None = None,
    date_range: tuple[int, int] | None = None,
    **kwargs,
) -> WorkflowResult:
    """Adapter for supervised_lit_review workflow."""
    from workflows.wrappers.supervised_lit_review.api import supervised_lit_review

    result = await supervised_lit_review(
        topic=query,
        research_questions=research_questions or [],
        quality=quality,
        language=language,
        date_range=date_range,
    )

    return WorkflowResult(
        final_report=result.get("final_report"),
        source_count=result.get("source_count", 0),
        status=result.get("status", "success" if result.get("final_report") else "failed"),
        errors=result.get("errors", []),
    )


def register_builtin_workflows() -> None:
    """Register all built-in workflows with the registry.

    Call this function to ensure all core workflows are available.
    This is idempotent - safe to call multiple times.
    """
    register_workflow(
        key="web_research",
        name="Web Research",
        runner=_run_web_research,
        default_enabled=True,
        requires_questions=False,
        supports_date_range=False,
        quality_tiers=["test", "quick", "standard", "comprehensive", "high_quality"],
        description="Deep web research using search and content scraping",
    )

    register_workflow(
        key="academic_lit_review",
        name="Academic Literature Review",
        runner=_run_academic_lit_review,
        default_enabled=False,
        requires_questions=True,
        supports_date_range=True,
        quality_tiers=["test", "quick", "standard", "comprehensive", "high_quality"],
        description="Academic paper discovery, processing, and synthesis",
    )

    register_workflow(
        key="book_finding",
        name="Book Finding",
        runner=_run_book_finding,
        default_enabled=False,
        requires_questions=False,
        supports_date_range=False,
        quality_tiers=["test", "quick", "standard", "comprehensive", "high_quality"],
        description="Book discovery and recommendation synthesis",
    )

    register_workflow(
        key="supervised_lit_review",
        name="Supervised Literature Review",
        runner=_run_supervised_lit_review,
        default_enabled=False,
        requires_questions=True,
        supports_date_range=True,
        quality_tiers=["test", "quick", "standard", "comprehensive", "high_quality"],
        description="Academic lit review with multi-loop supervision",
    )

    logger.debug(
        "Registered built-in workflows: web_research, academic_lit_review, "
        "book_finding, supervised_lit_review"
    )


# Note: Do NOT auto-register on import.
# Each wrapper workflow should explicitly call register_builtin_workflows()
# or register its own adapters (like multi_lang does).
# This allows different wrappers to use different workflow keys and adapters.
