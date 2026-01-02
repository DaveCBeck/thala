"""
Main entry point for book finding workflow.

Provides the public book_finding() function that orchestrates
the complete workflow from theme to markdown output.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from workflows.research.subgraphs.book_finding.state import (
    BookFindingInput,
    BookFindingState,
)
from .construction import book_finding_graph

logger = logging.getLogger(__name__)


async def book_finding(
    theme: str,
    brief: Optional[str] = None,
) -> dict[str, Any]:
    """Run book finding workflow for a theme.

    Generates book recommendations across three categories:
    1. Analogous domain - Books exploring theme from different fields
    2. Inspiring action - Books that inspire change and action
    3. Expressive fiction - Fiction capturing the theme's essence

    Each category uses Opus to generate 3 book recommendations,
    then the workflow searches for and processes the books via
    the book_search API and Marker PDF conversion.

    Args:
        theme: The theme to explore (e.g., "organizational resilience")
        brief: Optional additional context to guide recommendations

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
        )
        print(result["final_markdown"])
    """
    input_data = BookFindingInput(
        theme=theme,
        brief=brief,
    )

    initial_state = BookFindingState(
        input=input_data,
        analogous_recommendations=[],
        inspiring_recommendations=[],
        expressive_recommendations=[],
        search_results=[],
        processed_books=[],
        processing_failed=[],
        final_markdown=None,
        started_at=datetime.utcnow(),
        completed_at=None,
        current_phase="starting",
        errors=[],
    )

    logger.info(f"Starting book finding for theme: {theme[:100]}...")

    try:
        result = await book_finding_graph.ainvoke(initial_state)

        return {
            "final_markdown": result.get("final_markdown", ""),
            "processed_books": result.get("processed_books", []),
            "analogous_recommendations": result.get("analogous_recommendations", []),
            "inspiring_recommendations": result.get("inspiring_recommendations", []),
            "expressive_recommendations": result.get("expressive_recommendations", []),
            "search_results": result.get("search_results", []),
            "processing_failed": result.get("processing_failed", []),
            "started_at": initial_state["started_at"],
            "completed_at": result.get("completed_at"),
            "errors": result.get("errors", []),
        }

    except Exception as e:
        logger.error(f"Book finding workflow failed: {e}")
        return {
            "final_markdown": f"# Book Finding Failed\n\nError: {e}",
            "processed_books": [],
            "analogous_recommendations": [],
            "inspiring_recommendations": [],
            "expressive_recommendations": [],
            "search_results": [],
            "processing_failed": [],
            "started_at": initial_state["started_at"],
            "completed_at": datetime.utcnow(),
            "errors": [{"phase": "unknown", "error": str(e)}],
        }
