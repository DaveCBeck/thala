"""Finalize node for synthesis workflow."""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


async def finalize(state: dict) -> dict[str, Any]:
    """Finalize the synthesis workflow.

    Compiles final statistics, determines status, and prepares output.
    """
    final_report = state.get("final_report", "")
    errors = state.get("errors", [])
    web_research_results = state.get("web_research_results", [])
    book_finding_results = state.get("book_finding_results", [])

    # Determine status
    if not final_report:
        status = "failed"
    elif errors:
        status = "partial"
    else:
        status = "success"

    # Compile statistics
    web_sources = sum(r.get("source_count", 0) for r in web_research_results)
    books_processed = sum(
        len(r.get("processed_books", [])) for r in book_finding_results
    )

    logger.info(
        f"Synthesis finalized: status={status}, "
        f"report_length={len(final_report)}, "
        f"web_sources={web_sources}, "
        f"books={books_processed}, "
        f"errors={len(errors)}"
    )

    return {
        "status": status,
        "completed_at": datetime.utcnow(),
        "current_phase": "completed",
    }
