"""Illustration phase execution."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def run_illustration_phase(
    task: dict[str, Any],
    final_report: str,
    final_outputs: list[dict],
    get_output_dir_fn,
    generate_timestamp_fn,
    slugify_fn,
) -> dict[str, str]:
    """Run article illustration phase.

    Args:
        task: Task data
        final_report: Enhanced literature review
        final_outputs: Evening reads outputs
        get_output_dir_fn: Function to get output directory
        generate_timestamp_fn: Function to generate timestamp
        slugify_fn: Function to slugify strings

    Returns:
        Dict mapping article ID to illustrated file path
    """
    from workflows.output.illustrate import illustrate_graph
    from ..engines.illustration_engine import illustrate_articles

    logger.info("Phase 4: Illustrating articles")

    illustrated_paths = await illustrate_articles(
        task=task,
        final_report=final_report,
        final_outputs=final_outputs,
        illustrate_graph=illustrate_graph,
        get_output_dir_fn=get_output_dir_fn,
        generate_timestamp_fn=generate_timestamp_fn,
        slugify_fn=slugify_fn,
    )

    logger.info(f"Illustrated {len(illustrated_paths)} articles")

    return illustrated_paths
