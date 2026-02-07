"""Document illustration workflow.

Takes a markdown document and intelligently adds images using three sources:
- Public domain images (Pexels/Unsplash)
- AI-generated images (Imagen)
- SVG diagrams (Claude-generated)

Example:
    from workflows.output.illustrate import illustrate_document, IllustrateConfig

    result = await illustrate_document(
        markdown_document="# My Article\\n\\nContent here...",
        title="My Article",
        output_dir="/path/to/images",
        options=IllustrateConfig(
            additional_image_count=2,
            enable_vision_review=True,
        ),
    )

    illustrated_doc = result["illustrated_document"]
    images = result["final_images"]
"""

from typing import Any

from langsmith import traceable

from .config import IllustrateConfig
from .graph import illustrate_graph
from .schemas import DocumentAnalysis, ImageLocationPlan, VisionReviewResult
from .state import (
    FinalImage,
    IllustrateInput,
    IllustrateState,
    ImageGenResult,
    ImageReviewResult,
)


@traceable(run_type="chain", name="IllustrateDocument")
async def illustrate_document(
    markdown_document: str,
    title: str | None = None,
    output_dir: str | None = None,
    options: IllustrateConfig | None = None,
) -> dict[str, Any]:
    """Illustrate a markdown document with images.

    Args:
        markdown_document: Raw markdown content to illustrate
        title: Document title (extracted from content if not provided)
        output_dir: Directory to save image files (temp dir if not provided)
        options: Illustration configuration options

    Returns:
        Dict containing:
        - illustrated_document: Markdown with image references inserted
        - final_images: List of saved image metadata
        - status: "success", "partial", or "failed"
        - errors: Any errors encountered
    """
    result = await illustrate_graph.ainvoke({
        "input": {
            "markdown_document": markdown_document,
            "title": title,
            "output_dir": output_dir,
        },
        "config": options or IllustrateConfig(),
    })
    return result


__all__ = [
    # Main API
    "illustrate_document",
    # Graph (for direct access if needed)
    "illustrate_graph",
    # Config
    "IllustrateConfig",
    # State types
    "IllustrateState",
    "IllustrateInput",
    "ImageGenResult",
    "ImageReviewResult",
    "FinalImage",
    # Schemas
    "DocumentAnalysis",
    "ImageLocationPlan",
    "VisionReviewResult",
]
