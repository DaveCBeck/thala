"""Article illustration engine."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def illustrate_articles(
    task: dict[str, Any],
    final_report: str,
    final_outputs: list[dict],
    illustrate_graph,
    get_output_dir_fn,
    generate_timestamp_fn,
    slugify_fn,
) -> dict[str, str]:
    """Illustrate all articles with images.

    Args:
        task: Task data
        final_report: Enhanced literature review
        final_outputs: Evening reads outputs
        illustrate_graph: The illustrate workflow graph
        get_output_dir_fn: Function to get output directory
        generate_timestamp_fn: Function to generate timestamp
        slugify_fn: Function to slugify strings

    Returns:
        Dict mapping article ID to illustrated file path
    """
    output_dir = get_output_dir_fn()
    timestamp = generate_timestamp_fn()
    topic_slug = slugify_fn(task.get("topic", "unknown"))

    # Create illustration output directory
    illust_dir = output_dir / f"illustrated_{topic_slug}_{timestamp}"
    illust_dir.mkdir(exist_ok=True)

    illustrated_paths = {}

    # Illustrate the literature review
    try:
        lit_result = await illustrate_graph.ainvoke({
            "input": {
                "markdown_document": final_report,
                "title": f"Literature Review: {task.get('topic', 'Unknown')}",
                "output_dir": str(illust_dir / "lit_review_images"),
            }
        })

        lit_path = illust_dir / "lit_review.md"
        lit_content = lit_result.get("illustrated_document", final_report)
        lit_path.write_text(lit_content)
        illustrated_paths["lit_review"] = str(lit_path)

    except Exception as e:
        logger.error(f"Failed to illustrate lit review: {e}")
        # Save unillustrated version
        lit_path = illust_dir / "lit_review.md"
        lit_path.write_text(final_report)
        illustrated_paths["lit_review"] = str(lit_path)

    # Illustrate each evening reads article
    for output in final_outputs:
        article_id = output["id"]
        try:
            article_result = await illustrate_graph.ainvoke({
                "input": {
                    "markdown_document": output["content"],
                    "title": output["title"],
                    "output_dir": str(illust_dir / f"{article_id}_images"),
                }
            })

            article_path = illust_dir / f"{article_id}.md"
            article_content = article_result.get("illustrated_document", output["content"])
            article_path.write_text(article_content)
            illustrated_paths[article_id] = str(article_path)

        except Exception as e:
            logger.error(f"Failed to illustrate {article_id}: {e}")
            # Save unillustrated version
            article_path = illust_dir / f"{article_id}.md"
            article_path.write_text(output["content"])
            illustrated_paths[article_id] = str(article_path)

    return illustrated_paths
