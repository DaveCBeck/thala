"""Image generation node for evening_reads workflow.

Generates header images for all articles using LLM-generated prompts + Imagen.
"""

import asyncio
import logging
from typing import Any

from workflows.shared.image_utils import generate_article_header

from ..state import EveningReadsState, ImageOutput

logger = logging.getLogger(__name__)


async def generate_images_node(state: EveningReadsState) -> dict[str, Any]:
    """Generate header images for all articles.

    Uses a two-step process:
    1. Sonnet generates an optimized image prompt based on full article content
    2. Imagen generates the image from that prompt

    Generates images in parallel for:
    - 1 overview article
    - 3 deep-dive articles

    Returns:
        State update with image_outputs list
    """
    overview_draft = state.get("overview_draft")
    deep_dive_drafts = state.get("deep_dive_drafts", [])

    if not overview_draft and not deep_dive_drafts:
        logger.warning("No drafts available for image generation")
        return {"image_outputs": []}

    # Build image generation tasks
    tasks = []

    # Overview image
    if overview_draft:
        tasks.append(
            _generate_for_article(
                article_id="overview",
                title=overview_draft["title"],
                content=overview_draft["content"],
            )
        )

    # Deep-dive images
    for draft in deep_dive_drafts:
        tasks.append(
            _generate_for_article(
                article_id=draft["id"],
                title=draft["title"],
                content=draft["content"],
            )
        )

    # Run all image generations in parallel
    logger.info(f"Generating {len(tasks)} header images (Sonnet prompt + Imagen)...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful generations
    image_outputs = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Image generation task failed: {result}")
        elif result is not None:
            image_outputs.append(result)

    logger.info(
        f"Image generation complete: {len(image_outputs)}/{len(tasks)} successful"
    )

    return {"image_outputs": image_outputs}


async def _generate_for_article(
    article_id: str,
    title: str,
    content: str,
) -> ImageOutput | None:
    """Generate a header image for a single article.

    Args:
        article_id: Identifier for the article
        title: Article title
        content: Full article content (used by Sonnet to generate image prompt)

    Returns:
        ImageOutput if successful, None otherwise
    """
    image_bytes, prompt_used = await generate_article_header(
        title=title,
        content=content,
    )

    if image_bytes:
        return ImageOutput(
            article_id=article_id,
            image_bytes=image_bytes,
            prompt_used=prompt_used or "Unknown prompt",
        )

    logger.warning(f"Failed to generate image for {article_id}")
    return None
