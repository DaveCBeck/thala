"""Document illustration workflow.

Takes a markdown document and intelligently adds images using three sources:
- Public domain images (Pexels/Unsplash)
- AI-generated images (Imagen)
- SVG diagrams (Claude-generated)

Example:
    from workflows.output.illustrate import illustrate_graph, IllustrateConfig

    result = await illustrate_graph.ainvoke({
        "input": {
            "markdown_document": "# My Article\\n\\nContent here...",
            "title": "My Article",
            "output_dir": "/path/to/images",
        },
        "config": IllustrateConfig(
            additional_image_count=2,
            enable_vision_review=True,
        ),
    })

    illustrated_doc = result["illustrated_document"]
    images = result["final_images"]
"""

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

__all__ = [
    # Main graph
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
