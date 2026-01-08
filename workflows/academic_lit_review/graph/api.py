"""Main entry point for academic literature review workflow."""

import logging
from typing import Any, Optional

from workflows.academic_lit_review.state import (
    LitReviewInput,
    QUALITY_PRESETS,
)
from .state_init import build_initial_state
from .construction import academic_lit_review_graph

logger = logging.getLogger(__name__)


async def academic_lit_review(
    topic: str,
    research_questions: list[str],
    quality: str = "standard",
    language: str = "en",
    date_range: Optional[tuple[int, int]] = None,
    include_books: bool = True,
    focus_areas: Optional[list[str]] = None,
    exclude_terms: Optional[list[str]] = None,
    max_papers: Optional[int] = None,
) -> dict[str, Any]:
    """Run a complete academic literature review workflow.

    This is the main entry point for generating PhD-equivalent literature reviews.

    Args:
        topic: Research topic (e.g., "Large Language Models in Scientific Discovery")
        research_questions: List of specific questions to address
        quality: Quality tier - "quick", "standard", "comprehensive", "high_quality"
        language: ISO 639-1 language code (default: "en")
        date_range: Optional (start_year, end_year) filter
        include_books: Whether to include book sources (default: True)
        focus_areas: Optional specific areas to prioritize
        exclude_terms: Optional terms to filter out
        max_papers: Override default max papers for quality tier

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

    # Override max_papers if specified
    if max_papers:
        quality_settings["max_papers"] = max_papers

    # Build input
    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        date_range=date_range,
        include_books=include_books,
        focus_areas=focus_areas,
        exclude_terms=exclude_terms,
        max_papers=max_papers,
        language_code=language,
    )

    # Initialize state
    initial_state = build_initial_state(input_data, quality_settings)

    logger.info(f"Starting academic literature review: {topic}")
    logger.info(f"Quality: {quality}, Max papers: {quality_settings['max_papers']}")
    logger.info(f"Language: {language}")
    logger.info(f"LangSmith run ID: {initial_state['langsmith_run_id']}")

    try:
        result = await academic_lit_review_graph.ainvoke(initial_state)

        return {
            "final_review": result.get("final_review", ""),
            "paper_corpus": result.get("paper_corpus", {}),
            "paper_summaries": result.get("paper_summaries", {}),
            "clusters": result.get("clusters", []),
            "references": result.get("references", []),
            "citation_keys": list(result.get("zotero_keys", {}).values()),
            "zotero_keys": result.get("zotero_keys", {}),
            "elasticsearch_ids": result.get("elasticsearch_ids", {}),
            "prisma_documentation": result.get("prisma_documentation", ""),
            "diffusion": result.get("diffusion", {}),
            "quality_metrics": result.get("section_drafts", {}).get("quality_metrics"),
            "started_at": initial_state["started_at"],
            "completed_at": result.get("completed_at"),
            "langsmith_run_id": initial_state["langsmith_run_id"],
            "errors": result.get("errors", []),
        }

    except Exception as e:
        logger.error(f"Literature review failed: {e}")
        return {
            "final_review": f"Literature review generation failed: {e}",
            "paper_corpus": {},
            "paper_summaries": {},
            "clusters": [],
            "references": [],
            "langsmith_run_id": initial_state["langsmith_run_id"],
            "errors": [{"phase": "unknown", "error": str(e)}],
        }
