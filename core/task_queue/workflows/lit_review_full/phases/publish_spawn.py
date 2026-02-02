"""Publish task spawning phase execution."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def run_publish_spawn_phase(
    task: dict[str, Any],
    lit_review_path: Optional[str],
    illustrated_paths: dict[str, str],
    final_outputs: list[dict],
) -> str:
    """Spawn publish_series task.

    Args:
        task: Parent task data
        lit_review_path: Path to illustrated lit review
        illustrated_paths: Map of article ID to illustrated file path
        final_outputs: Evening reads outputs

    Returns:
        New task ID
    """
    from ..engines.publish_builder import spawn_publish_task

    logger.info("Phase 5: Spawning publish_series task")

    publish_task_id = spawn_publish_task(
        task=task,
        lit_review_path=lit_review_path,
        illustrated_paths=illustrated_paths,
        final_outputs=final_outputs,
    )

    logger.info(f"Spawned publish_series task: {publish_task_id}")

    return publish_task_id
