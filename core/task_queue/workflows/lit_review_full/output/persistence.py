"""Output persistence for lit_review_full workflow."""

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def save_workflow_outputs(
    task: dict[str, Any],
    result: dict[str, Any],
    get_output_dir_fn,
    generate_timestamp_fn,
    slugify_fn,
) -> dict[str, str]:
    """Save literature review and article series to .thala/output/ directory.

    Note: Illustrated versions are already saved during the illustrate phase.
    This method saves additional metadata and non-illustrated versions.

    Args:
        task: Task data
        result: Workflow result
        get_output_dir_fn: Function to get output directory
        generate_timestamp_fn: Function to generate timestamp
        slugify_fn: Function to slugify strings

    Returns:
        Dict with paths to saved files
    """
    output_paths = {}

    # If we have illustrated paths, use those as the primary outputs
    illustrated_paths = result.get("illustrated_paths", {})
    if illustrated_paths:
        output_paths.update(illustrated_paths)
        logger.info(f"Using illustrated paths: {list(illustrated_paths.keys())}")

    # Save metadata summary
    output_dir = get_output_dir_fn()
    timestamp = generate_timestamp_fn()
    topic_slug = slugify_fn(task.get("topic", "unknown"))

    # Save a summary file with all paths and publish task info
    summary_path = output_dir / f"summary_{topic_slug}_{timestamp}.json"
    summary = {
        "topic": task.get("topic"),
        "quality": task.get("quality"),
        "category": task.get("category"),
        "generated_at": datetime.now().isoformat(),
        "illustrated_paths": illustrated_paths,
        "publish_task_id": result.get("publish_task_id"),
        "status": result.get("status"),
        "errors": result.get("errors"),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    output_paths["summary"] = str(summary_path)

    # If no illustrated versions, save raw versions
    if not illustrated_paths:
        lit_result = result.get("lit_review", {})
        final_report = result.get("final_report") or lit_result.get("final_report")

        if final_report:
            lit_review_path = output_dir / f"lit_review_{topic_slug}_{timestamp}.md"
            with open(lit_review_path, "w") as f:
                f.write(f"# Literature Review: {task.get('topic', 'Unknown')}\n\n")
                f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write(f"*Quality: {task.get('quality', 'standard')}*\n\n")
                if task.get("research_questions"):
                    f.write("## Research Questions\n\n")
                    for q in task["research_questions"]:
                        f.write(f"- {q}\n")
                    f.write("\n---\n\n")
                f.write(final_report)

            output_paths["lit_review"] = str(lit_review_path)
            logger.info(f"Saved lit review to {lit_review_path}")

        # Save article series
        series_result = result.get("series", {})
        final_outputs = series_result.get("final_outputs", [])

        if final_outputs:
            series_dir = output_dir / f"series_{topic_slug}_{timestamp}"
            series_dir.mkdir(exist_ok=True)
            output_paths["series_dir"] = str(series_dir)

            for output in final_outputs:
                article_path = series_dir / f"{output['id']}.md"
                with open(article_path, "w") as f:
                    f.write(f"# {output['title']}\n\n")
                    f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                    f.write("---\n\n")
                    f.write(output["content"])
                output_paths[output["id"]] = str(article_path)

            logger.info(f"Saved {len(final_outputs)} articles to {series_dir}")

    return output_paths
