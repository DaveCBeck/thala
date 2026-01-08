"""
Built-in workflow adapters for multi-language research.

This module defines the adapter functions for the three core workflows:
- web: Deep web research using deep_research
- academic: Academic literature review using academic_lit_review
- books: Book finding using book_finding

These are registered automatically when the multi_lang module is imported.
"""

import logging
from workflows.multi_lang.workflow_registry import (
    register_workflow,
    WorkflowResult,
)

logger = logging.getLogger(__name__)


async def run_web_research(
    topic: str,
    language_config: dict,
    quality: str,
    research_questions: list[str] | None = None,
) -> WorkflowResult:
    """Adapter for deep_research workflow.

    Args:
        topic: The research topic
        language_config: Dict with code, name, locale, preferred_domains
        quality: Quality level (quick, standard, comprehensive)
        research_questions: Optional research questions (not used by deep_research)

    Returns:
        Normalized WorkflowResult dict
    """
    from workflows.research.graph.api import deep_research

    try:
        result = await deep_research(
            query=topic,
            depth=quality,
            language=language_config["code"],
        )

        return WorkflowResult(
            final_report=result.get("final_report"),
            source_count=len(result.get("research_findings", [])),
            status="completed" if result.get("final_report") else "failed",
            errors=result.get("errors", []),
        )
    except Exception as e:
        logger.error(f"Web research failed for {language_config['code']}: {e}")
        return WorkflowResult(
            final_report=None,
            source_count=0,
            status="failed",
            errors=[{"phase": "web_research", "error": str(e)}],
        )


async def run_academic_research(
    topic: str,
    language_config: dict,
    quality: str,
    research_questions: list[str] | None = None,
) -> WorkflowResult:
    """Adapter for academic_lit_review workflow.

    Args:
        topic: The research topic
        language_config: Dict with code, name, locale, preferred_domains
        quality: Quality level (test, quick, standard, comprehensive, high_quality)
        research_questions: Research questions to guide the review

    Returns:
        Normalized WorkflowResult dict
    """
    from workflows.academic_lit_review.graph.api import (
        academic_lit_review,
    )

    try:
        # Default questions if not provided
        questions = research_questions or [
            f"What is the current state of research on {topic}?",
            f"What are the key findings and debates about {topic}?",
        ]

        result = await academic_lit_review(
            topic=topic,
            research_questions=questions,
            quality=quality,
            language=language_config["code"],
        )

        return WorkflowResult(
            final_report=result.get("final_review"),
            source_count=len(result.get("paper_corpus", {})),
            status="completed" if result.get("final_review") else "failed",
            errors=result.get("errors", []),
        )
    except Exception as e:
        logger.error(f"Academic research failed for {language_config['code']}: {e}")
        return WorkflowResult(
            final_report=None,
            source_count=0,
            status="failed",
            errors=[{"phase": "academic_research", "error": str(e)}],
        )


async def run_book_research(
    topic: str,
    language_config: dict,
    quality: str,
    research_questions: list[str] | None = None,
) -> WorkflowResult:
    """Adapter for book_finding workflow.

    Args:
        topic: The research topic/theme
        language_config: Dict with code, name, locale, preferred_domains
        quality: Quality level (quick, standard, comprehensive)
        research_questions: Not used by book_finding

    Returns:
        Normalized WorkflowResult dict
    """
    from workflows.book_finding.graph.api import book_finding

    try:
        result = await book_finding(
            theme=topic,
            quality=quality,
            language=language_config["code"],
        )

        return WorkflowResult(
            final_report=result.get("final_markdown"),
            source_count=len(result.get("processed_books", [])),
            status="completed" if result.get("final_markdown") else "failed",
            errors=result.get("errors", []),
        )
    except Exception as e:
        logger.error(f"Book research failed for {language_config['code']}: {e}")
        return WorkflowResult(
            final_report=None,
            source_count=0,
            status="failed",
            errors=[{"phase": "book_research", "error": str(e)}],
        )


def register_builtin_workflows() -> None:
    """Register all built-in workflows with the registry.

    This is called automatically when the multi_lang module is imported.
    """
    register_workflow(
        key="web",
        name="Web Research",
        runner=run_web_research,
        default_enabled=True,
        requires_questions=False,  # deep_research doesn't use research_questions
        description="Deep web research using search and content scraping",
    )

    register_workflow(
        key="academic",
        name="Academic Literature Review",
        runner=run_academic_research,
        default_enabled=False,
        requires_questions=True,
        description="Academic paper discovery, processing, and synthesis",
    )

    register_workflow(
        key="books",
        name="Book Finding",
        runner=run_book_research,
        default_enabled=False,
        requires_questions=False,
        description="Book discovery and recommendation synthesis",
    )

    logger.debug("Registered built-in workflows: web, academic, books")
