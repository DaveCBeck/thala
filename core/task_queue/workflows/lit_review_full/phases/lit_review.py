"""Literature review phase execution."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def run_lit_review_phase(
    topic: str,
    research_questions: Optional[list[str]],
    quality: str,
    language: str,
    date_range: Optional[dict],
) -> dict[str, Any]:
    """Run academic literature review phase.

    Args:
        topic: Research topic
        research_questions: Optional research questions
        quality: Quality level
        language: Language code
        date_range: Optional date range filter

    Returns:
        Literature review result with final_report and paper_corpus
    """
    from workflows.research.academic_lit_review import academic_lit_review

    logger.info("Phase 1: Running academic literature review")

    lit_result = await academic_lit_review(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        language=language,
        date_range=date_range,
    )

    if not lit_result.get("final_report"):
        raise RuntimeError(
            f"Literature review failed: {lit_result.get('errors', 'Unknown error')}"
        )

    logger.info(
        f"Lit review complete: {len(lit_result.get('paper_corpus', {}))} papers"
    )

    return lit_result
