"""Main entry point for academic literature review workflow."""

import logging
import uuid
from typing import Any, Optional

from workflows.research.academic_lit_review.state import LitReviewInput
from workflows.research.academic_lit_review.quality_presets import QUALITY_PRESETS
from workflows.shared.quality_config import QualityTier
from workflows.shared.workflow_state_store import save_workflow_state
from .state_init import build_initial_state
from .construction import academic_lit_review_graph

logger = logging.getLogger(__name__)


async def academic_lit_review(
    topic: str,
    research_questions: list[str],
    quality: QualityTier = "standard",
    language: str = "en",
    date_range: Optional[tuple[int, int]] = None,
) -> dict[str, Any]:
    """Run a complete academic literature review workflow.

    This is the main entry point for generating PhD-equivalent literature reviews.

    Args:
        topic: Research topic (e.g., "Large Language Models in Scientific Discovery")
        research_questions: List of specific questions to address
        quality: Quality tier - "quick", "standard", "comprehensive", "high_quality"
        language: ISO 639-1 language code (default: "en")
        date_range: Optional (start_year, end_year) filter

    Returns:
        Dict containing:
        - final_review: Complete literature review text with citations
        - paper_corpus: All discovered papers
        - paper_summaries: Processed paper summaries
        - clusters: Thematic clusters
        - references: Formatted citations
        - prisma_documentation: Search methodology docs
        - quality_metrics: Review quality metrics
        - errors: Any errors encountered

    Example:
        result = await academic_lit_review(
            topic="Large Language Models in Scientific Discovery",
            research_questions=[
                "How are LLMs being used for hypothesis generation?",
                "What are the methodological challenges of using LLMs in research?",
            ],
            quality="high_quality",
            language="es",
            date_range=(2020, 2025),
        )

        # Access results
        print(f"Papers analyzed: {len(result['paper_corpus'])}")
        print(f"Review length: {len(result['final_review'].split())} words")

        # Save to file
        with open("literature_review.md", "w") as f:
            f.write(result['final_review'])
    """
    # Get quality settings
    if quality not in QUALITY_PRESETS:
        logger.warning(f"Unknown quality '{quality}', using 'standard'")
        quality = "standard"

    quality_settings = QUALITY_PRESETS[quality].copy()

    # Build input
    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        date_range=date_range,
        language_code=language,
    )

    # Initialize state
    initial_state = build_initial_state(input_data, quality_settings)

    logger.info(
        f"Starting academic literature review: '{topic}' "
        f"(quality={quality}, max_papers={quality_settings['max_papers']}, language={language})"
    )
    logger.debug(f"LangSmith run ID: {initial_state['langsmith_run_id']}")

    try:
        run_id = uuid.UUID(initial_state["langsmith_run_id"])
        result = await academic_lit_review_graph.ainvoke(
            initial_state,
            config={
                "run_id": run_id,
                "run_name": f"lit_review:{topic[:30]}",
            },
        )

        final_review = result.get("final_review", "")
        errors = result.get("errors", [])

        # Determine standardized status
        if final_review and not errors:
            status = "success"
        elif final_review and errors:
            status = "partial"
        else:
            status = "failed"

        # Save full state for downstream workflows (in dev/test mode)
        save_workflow_state(
            workflow_name="academic_lit_review",
            run_id=initial_state["langsmith_run_id"],
            state={
                "input": dict(input_data)
                if hasattr(input_data, "_asdict")
                else input_data,
                "paper_corpus": result.get("paper_corpus", {}),
                "paper_summaries": result.get("paper_summaries", {}),
                "clusters": result.get("clusters", []),
                "zotero_keys": result.get("zotero_keys", {}),
                "elasticsearch_ids": result.get("elasticsearch_ids", {}),
                "references": result.get("references", []),
                "diffusion": result.get("diffusion", {}),
                "final_review": final_review,
                "quality_settings": quality_settings,
                "started_at": initial_state["started_at"],
                "completed_at": result.get("completed_at"),
            },
        )

        return {
            "final_report": final_review,
            "status": status,
            "langsmith_run_id": initial_state["langsmith_run_id"],
            "errors": errors,
            "source_count": len(result.get("paper_corpus", {})),
            "started_at": initial_state["started_at"],
            "completed_at": result.get("completed_at"),
        }

    except Exception as e:
        logger.error(f"Literature review failed: {e}")
        return {
            "final_report": f"Literature review generation failed: {e}",
            "status": "failed",
            "langsmith_run_id": initial_state["langsmith_run_id"],
            "errors": [{"phase": "unknown", "error": str(e)}],
            "source_count": 0,
            "started_at": initial_state["started_at"],
            "completed_at": None,
        }
