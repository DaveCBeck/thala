"""Configuration for document illustration workflow."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .schemas import VisualIdentity


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
        default=3,
        ge=0,
        le=5,
        description="Number of additional images beyond header (default 3)",
    )

    # Cost control
    imagen_sample_count: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of Imagen candidates per generation call",
    )
    overgeneration_surplus: int = Field(
        default=2,
        ge=0,
        le=2,
        description="Extra image locations beyond target for editorial curation",
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
        default=900,
        description="Width for SVG diagrams in pixels",
    )
    diagram_height: int = Field(
        default=600,
        description="Height for SVG diagrams in pixels",
    )

    # Editorial review settings
    enable_editorial_review: bool = Field(
        default=True,
        description="Enable full-document editorial review to cut weakest images",
    )

    # Retry settings
    max_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Max retry rounds for locations where both candidates fail",
    )

    # Visual identity caching (series reuse)
    visual_identity_override: VisualIdentity | None = Field(
        default=None,
        description="Pre-established visual identity to reuse across a series",
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

    # --- Presets -------------------------------------------------------

    @classmethod
    def quick(cls, **overrides) -> IllustrateConfig:
        """Minimal cost: no surplus, no editorial review, no retries."""
        defaults = dict(
            overgeneration_surplus=0,
            enable_editorial_review=False,
            max_retries=0,
            imagen_sample_count=1,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def balanced(cls, **overrides) -> IllustrateConfig:
        """Moderate cost: surplus=1, single retry."""
        defaults = dict(
            overgeneration_surplus=1,
            enable_editorial_review=True,
            max_retries=1,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def quality(cls, **overrides) -> IllustrateConfig:
        """Full quality: surplus=2, 2 Imagen samples."""
        defaults = dict(
            overgeneration_surplus=2,
            imagen_sample_count=2,
        )
        defaults.update(overrides)
        return cls(**defaults)
