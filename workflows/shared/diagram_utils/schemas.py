"""Schema definitions for diagram generation.

Contains all type definitions, enums, Pydantic models, and dataclasses
used throughout the diagram generation pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class DiagramType(str, Enum):
    """Supported diagram types."""

    FLOWCHART = "flowchart"
    CONCEPT_MAP = "concept_map"
    PROCESS_DIAGRAM = "process_diagram"
    HIERARCHY = "hierarchy"
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    CYCLE = "cycle"


class DiagramAnalysis(BaseModel):
    """Result of analyzing content for diagram generation."""

    should_generate: bool = Field(
        description="Whether a diagram would meaningfully enhance this content"
    )
    diagram_type: DiagramType = Field(
        description="Most appropriate diagram type for this content"
    )
    title: str = Field(description="Concise title for the diagram (3-8 words)")
    key_elements: list[str] = Field(
        description="Key concepts/entities to include in the diagram (maximum 15, typically 5-10)",
        min_length=1,
        max_length=15,
    )
    relationships: list[str] = Field(
        description="Key relationships or flows between elements"
    )
    rationale: str = Field(
        description="Brief explanation of why this diagram type fits the content"
    )


class DiagramConfig(BaseModel):
    """Configuration for diagram generation."""

    width: int = Field(default=800, description="SVG width in pixels")
    height: int = Field(default=600, description="SVG height in pixels")
    dpi: int = Field(default=150, description="DPI for PNG conversion")
    background_color: str = Field(default="#ffffff", description="Background color")
    font_family: str = Field(default="Arial, sans-serif", description="Font family")
    primary_color: str = Field(default="#2563eb", description="Primary accent color")
    num_candidates: int = Field(
        default=3, ge=1, le=10, description="Number of SVG candidates to generate in parallel"
    )

    # Refinement loop settings
    enable_refinement_loop: bool = Field(
        default=False,
        description="Enable iterative quality refinement loop",
    )
    max_refinement_iterations: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum refinement iterations",
    )
    quality_threshold: float = Field(
        default=4.7,
        ge=1.0,
        le=5.0,
        description="Minimum quality score to accept (1-5 scale)",
    )


class OverlapCheckResult(BaseModel):
    """Result of text overlap validation."""

    has_overlaps: bool = Field(description="Whether any text elements overlap")
    overlap_pairs: list[tuple[str, str]] = Field(
        default_factory=list, description="Pairs of overlapping text labels"
    )
    text_shape_overlaps: list[str] = Field(
        default_factory=list,
        description="Text labels that overlap with shapes (e.g., circles, dots)",
    )
    suggestion: Optional[str] = Field(
        default=None, description="Suggestion for fixing overlaps if found"
    )


class BoundsCheckResult(BaseModel):
    """Result of checking if elements exceed SVG bounds."""

    has_violations: bool = Field(description="Whether any elements exceed bounds")
    violations: list[str] = Field(
        default_factory=list,
        description="Descriptions of elements that exceed bounds",
    )
    svg_width: float = Field(default=800, description="SVG viewBox width")
    svg_height: float = Field(default=600, description="SVG viewBox height")


class QualityIssue(BaseModel):
    """A specific quality issue identified in a diagram."""

    category: str = Field(description="Category of the quality issue")
    severity: Literal["minor", "moderate", "severe"] = Field(
        description="How serious the issue is"
    )
    description: str = Field(description="Description of the issue")
    affected_elements: list[str] = Field(
        default_factory=list,
        description="Elements affected by this issue",
    )
    suggested_fix: str = Field(description="Suggested fix for the issue")

    @property
    def normalized_category(self) -> str:
        """Return normalized category name."""
        # Map alternative names to canonical names
        category_map = {
            "visual_hierarchy": "hierarchy",
            "text_legibility": "text_legibility",
            "overlap_free": "overlap",
            "overlap-free": "overlap",
            "spacing_balance": "spacing",
            "layout_logic": "layout",
            "shape_appropriateness": "shape",
        }
        return category_map.get(self.category, self.category)


class DiagramQualityAssessment(BaseModel):
    """Vision model's assessment of diagram visual quality."""

    # Individual scores (1-5)
    text_legibility: int = Field(
        ge=1, le=5, description="Text readability, font sizes, contrast"
    )
    overlap_free: int = Field(
        ge=1, le=5, description="Absence of unintended overlaps"
    )
    visual_hierarchy: int = Field(
        ge=1, le=5, description="Clear distinction between primary/secondary elements"
    )
    spacing_balance: int = Field(
        ge=1, le=5, description="Even whitespace distribution, adequate margins"
    )
    layout_logic: int = Field(
        ge=1, le=5, description="Elements arranged logically for the diagram type"
    )
    shape_appropriateness: int = Field(
        ge=1, le=5, description="Shapes match semantic meaning, consistent styling"
    )
    completeness: int = Field(
        ge=1, le=5, description="All key elements and relationships represented"
    )

    # Overall
    overall_score: float = Field(
        ge=1.0, le=5.0, description="Weighted average of all criteria"
    )
    issues: list[QualityIssue] = Field(
        default_factory=list, description="Specific issues identified"
    )
    meets_threshold: bool = Field(
        description="Whether the diagram meets the quality threshold"
    )


@dataclass
class DiagramCandidate:
    """A candidate SVG with its quality metrics."""

    svg_content: str
    png_bytes: bytes
    overlap_check: OverlapCheckResult
    candidate_id: int  # 1, 2, or 3


@dataclass
class DiagramResult:
    """Result of diagram generation."""

    svg_bytes: bytes | None
    png_bytes: bytes | None
    analysis: DiagramAnalysis | None
    overlap_check: OverlapCheckResult | None
    generation_attempts: int  # Number of candidates generated
    selected_candidate: int | None = None  # Which candidate was chosen (1, 2, or 3)
    selection_rationale: str | None = None  # Why it was chosen
    improvements_made: list[str] | None = None  # What was improved
    success: bool = False
    error: str | None = None
    source_code: str | None = None  # Raw source code (Mermaid/DOT) when engine is not SVG-native

    # Refinement metrics (if refinement loop was enabled)
    refinement_iterations: int | None = None
    final_quality_score: float | None = None
    quality_history: list[float] | None = None  # Score at each iteration

    @classmethod
    def failure(cls, error: str) -> DiagramResult:
        """Create a failed DiagramResult."""
        return cls(
            svg_bytes=None,
            png_bytes=None,
            analysis=None,
            overlap_check=None,
            generation_attempts=1,
            success=False,
            error=error,
        )


logger = logging.getLogger(__name__)

# Type alias for diagram generator functions (e.g. generate_mermaid_diagram,
# generate_graphviz_diagram).
GeneratorFn = Callable[
    [str, DiagramConfig, str],
    Coroutine[Any, Any, DiagramResult],
]


async def generate_with_selection(
    generator_fn: GeneratorFn,
    analysis: str,
    config: DiagramConfig,
    custom_instructions: str = "",
    num_candidates: int = 3,
    engine_name: str = "diagram",
) -> DiagramResult:
    """Generate multiple diagram candidates, select best via vision.

    Shared logic for mermaid and graphviz selection flows.

    Args:
        generator_fn: Async function that generates a single DiagramResult.
        analysis: Content analysis or description.
        config: Diagram configuration.
        custom_instructions: Additional instructions for generation.
        num_candidates: Number of parallel candidates to generate.
        engine_name: Human-readable engine name for error messages.

    Returns:
        Best DiagramResult from candidates.
    """
    from workflows.shared.vision_comparison import vision_pair_select

    instructions = custom_instructions or analysis

    tasks = [generator_fn(instructions, config, custom_instructions) for _ in range(num_candidates)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = [r for r in results if isinstance(r, DiagramResult) and r.success and r.png_bytes]

    if not successful:
        return DiagramResult.failure(f"All {engine_name} candidates failed")
    if len(successful) == 1:
        return successful[0]

    # Vision pair comparison to select best
    png_list = [r.png_bytes for r in successful]
    try:
        best_idx = await vision_pair_select(png_list, instructions)
    except Exception as e:
        logger.warning(f"Vision selection failed, using first candidate: {e}")
        best_idx = 0

    selected = successful[best_idx]
    selected.selected_candidate = best_idx + 1
    selected.generation_attempts = len(successful)
    return selected


__all__ = [
    "DiagramType",
    "DiagramAnalysis",
    "DiagramConfig",
    "OverlapCheckResult",
    "BoundsCheckResult",
    "QualityIssue",
    "DiagramQualityAssessment",
    "DiagramCandidate",
    "DiagramResult",
    "generate_with_selection",
]
