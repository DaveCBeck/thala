"""Evening reads phase execution."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def run_evening_reads_phase(
    final_report: str,
    category: Optional[str] = None,
) -> dict[str, Any]:
    """Run evening reads article series generation phase.

    Args:
        final_report: Enhanced literature review
        category: Optional category for editorial stance

    Returns:
        Series result with final_outputs
    """
    from workflows.output.evening_reads import evening_reads_graph
    from workflows.output.evening_reads.editorial import load_editorial_stance

    logger.info("Phase 3: Generating evening reads series")

    # Load editorial stance for the publication
    editorial_stance = load_editorial_stance(category or "")
    if editorial_stance:
        logger.info(f"Using editorial stance for category: {category}")

    series_result = await evening_reads_graph.ainvoke({
        "input": {
            "literature_review": final_report,
            "editorial_stance": editorial_stance,
        }
    })

    if not series_result.get("final_outputs"):
        raise RuntimeError(
            f"Series generation failed: {series_result.get('errors', 'Unknown error')}"
        )

    logger.info(
        f"Series complete: {len(series_result.get('final_outputs', []))} articles"
    )

    return series_result
