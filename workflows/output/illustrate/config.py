"""Configuration for document illustration workflow."""

from pydantic import BaseModel, Field


class IllustrateConfig(BaseModel):
    """Configuration for document illustration.

    Controls image generation behavior, counts, and quality settings.
    """

    # Image count settings
    generate_header_image: bool = Field(
        default=True,
        description="Whether to generate a header image",
    )
    additional_image_count: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Number of additional images beyond header (default 2)",
    )

    # Header image preference
    header_prefer_public_domain: bool = Field(
        default=True,
        description="Prefer public domain for header if 'particularly apposite'",
    )

    # Image generation settings
    imagen_aspect_ratio: str = Field(
        default="16:9",
        description="Aspect ratio for Imagen-generated images",
    )
    diagram_width: int = Field(
        default=800,
        description="Width for SVG diagrams in pixels",
    )
    diagram_height: int = Field(
        default=600,
        description="Height for SVG diagrams in pixels",
    )

    # Review settings
    enable_vision_review: bool = Field(
        default=True,
        description="Enable Sonnet vision review of generated images",
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Max retries for substantive errors (one-retry-then-fail-through)",
    )

    # Output settings
    output_dir: str | None = Field(
        default=None,
        description="Directory to save image files. If None, uses temp directory.",
    )

    # Diagram refinement settings
    enable_diagram_refinement: bool = Field(
        default=True,
        description="Enable iterative quality refinement for diagrams",
    )
    diagram_quality_threshold: float = Field(
        default=4.7,
        ge=1.0,
        le=5.0,
        description="Minimum quality score for diagrams (1-5 scale)",
    )
    diagram_max_refinement_iterations: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum refinement iterations for diagrams",
    )
