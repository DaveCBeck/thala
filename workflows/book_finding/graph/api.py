"""
Main entry point for book finding workflow.

Provides the public book_finding() function that orchestrates
the complete workflow from theme to markdown output.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from workflows.shared.language import get_language_config
from workflows.shared.workflow_state_store import save_workflow_state
from workflows.book_finding.state import (
    BookFindingInput,
    BookFindingState,
    QUALITY_PRESETS,
)
from .construction import book_finding_graph

logger = logging.getLogger(__name__)


async def book_finding(
    theme: str,
    brief: Optional[str] = None,
    quality: Literal["quick", "standard", "comprehensive"] = "standard",
    language: str = "en",
) -> dict[str, Any]:
    """Run book finding workflow for a theme.

    Generates book recommendations across three categories:
    1. Analogous domain - Books exploring theme from different fields
    2. Inspiring action - Books that inspire change and action
    3. Expressive fiction - Fiction capturing the theme's essence

    Each category generates book recommendations (2-5 per category depending
    on quality tier), then the workflow searches for and processes the books
    via the book_search API and Marker PDF conversion.

    Args:
        theme: The theme to explore (e.g., "organizational resilience")
        brief: Optional additional context to guide recommendations
        quality: Quality tier - "quick", "standard", or "comprehensive"
            - quick: 2 recommendations per category, faster processing
            - standard: 3 recommendations per category (default)
            - comprehensive: 5 recommendations per category, thorough
        language: ISO 639-1 language code (default: "en")

    Returns:
        Dict containing:
        - final_markdown: Complete markdown document with all recommendations
        - processed_books: List of books that were successfully processed
        - analogous_recommendations: Analogous domain recommendations
        - inspiring_recommendations: Inspiring action recommendations
        - expressive_recommendations: Expressive fiction recommendations
        - processing_failed: List of book titles that failed to process
        - errors: Any errors encountered during workflow

    Example:
        result = await book_finding(
            theme="creative leadership in uncertain times",
            brief="Focus on practical approaches rather than theoretical frameworks",
            quality="comprehensive",
            language="es",
        )
        print(result["final_markdown"])
    """
    # Get quality settings, defaulting to standard if invalid
    quality_settings = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])

    # Get language configuration
    language_config = get_language_config(language)

    input_data = BookFindingInput(
        theme=theme,
        brief=brief,
        quality=quality,
        language_code=language,
    )

    initial_state = BookFindingState(
        input=input_data,
        quality_settings=quality_settings,
        language_config=language_config,
        analogous_recommendations=[],
        inspiring_recommendations=[],
        expressive_recommendations=[],
        search_results=[],
        processed_books=[],
        processing_failed=[],
        final_markdown=None,
        final_report=None,
        started_at=datetime.utcnow(),
        completed_at=None,
        current_phase="starting",
        status=None,
        errors=[],
    )

    run_id = uuid.uuid4()
    logger.info(f"Starting book finding workflow for theme '{theme[:100]}' (quality: {quality}, language: {language})")
    logger.debug(f"LangSmith run ID: {run_id}")

    try:
        result = await book_finding_graph.ainvoke(
            initial_state,
            config={
                "run_id": run_id,
                "run_name": f"books:{theme[:30]}",
            },
        )

        final_markdown = result.get("final_markdown", "")
        errors = result.get("errors", [])

        # Determine standardized status
        if final_markdown and not errors:
            status = "success"
        elif final_markdown and errors:
            status = "partial"
        else:
            status = "failed"

        # Save full state for downstream workflows (in dev/test mode)
        save_workflow_state(
            workflow_name="book_finding",
            run_id=str(run_id),
            state={
                "input": dict(input_data) if hasattr(input_data, "_asdict") else input_data,
                "processed_books": result.get("processed_books", []),
                "analogous_recommendations": result.get("analogous_recommendations", []),
                "inspiring_recommendations": result.get("inspiring_recommendations", []),
                "expressive_recommendations": result.get("expressive_recommendations", []),
                "search_results": result.get("search_results", []),
                "processing_failed": result.get("processing_failed", []),
                "final_markdown": final_markdown,
                "started_at": initial_state["started_at"],
                "completed_at": result.get("completed_at"),
            },
        )

        logger.info(f"Book finding workflow completed with status '{status}' ({len(result.get('processed_books', []))} books processed)")

        return {
            "final_report": final_markdown,
            "status": status,
            "langsmith_run_id": str(run_id),
            "errors": errors,
            "source_count": len(result.get("processed_books", [])),
            "started_at": initial_state["started_at"],
            "completed_at": result.get("completed_at"),
        }

    except Exception as e:
        logger.error(f"Book finding workflow failed: {e}")
        return {
            "final_report": f"# Book Finding Failed\n\nError: {e}",
            "status": "failed",
            "langsmith_run_id": str(run_id),
            "errors": [{"phase": "unknown", "error": str(e)}],
            "source_count": 0,
            "started_at": initial_state["started_at"],
            "completed_at": datetime.utcnow(),
        }
