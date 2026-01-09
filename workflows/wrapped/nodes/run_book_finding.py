"""
Book finding node.

Runs the book finding workflow with the generated theme and brief.
"""

import logging
from datetime import datetime
from typing import Any

from workflows.book_finding import book_finding
from workflows.wrapped.state import WrappedResearchState, WorkflowResult, QUALITY_MAPPING

logger = logging.getLogger(__name__)


async def run_book_finding(state: WrappedResearchState) -> dict[str, Any]:
    """Run book finding workflow with generated theme.

    Uses the theme and brief generated from web/academic research
    to find relevant books.
    """
    theme = state.get("book_theme") or state["input"]["query"]
    brief = state.get("book_brief")

    # Get quality setting for book finding
    quality = state["input"]["quality"]
    book_quality = QUALITY_MAPPING[quality]["book_quality"]

    started_at = datetime.utcnow()

    try:
        logger.info(f"Starting book finding for theme: {theme[:100]}... (quality: {book_quality})")

        result = await book_finding(
            theme=theme,
            brief=brief,
            quality=book_quality,
        )

        book_result = WorkflowResult(
            workflow_type="books",
            final_output=result.get("final_report"),  # Use standardized field
            started_at=started_at,
            completed_at=datetime.utcnow(),
            status=result.get("status", "completed"),
            error=None,
            top_of_mind_id=None,
        )

        logger.info(
            f"Book finding complete. "
            f"Processed: {len(result.get('processed_books', []))}, "
            f"Failed: {len(result.get('processing_failed', []))}"
        )

        return {
            "book_result": book_result,
            "current_phase": "book_finding_complete",
        }

    except Exception as e:
        logger.error(f"Book finding failed: {e}")
        return {
            "book_result": WorkflowResult(
                workflow_type="books",
                final_output=None,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                status="failed",
                error=str(e),
                top_of_mind_id=None,
            ),
            "current_phase": "book_finding_complete",
            "errors": [{"phase": "book_finding", "error": str(e)}],
        }
