"""Schema definitions for diagram generation.

Contains all type definitions, enums, Pydantic models, and dataclasses
used throughout the diagram generation pipeline.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

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
        description="3-8 key concepts/entities to include in the diagram",
        min_length=1,
        max_length=10,
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
        default=3, description="Number of SVG candidates to generate in parallel"
    )


class OverlapCheckResult(BaseModel):
    """Result of text overlap validation."""

    has_overlaps: bool = Field(description="Whether any text elements overlap")
    overlap_pairs: list[tuple[str, str]] = Field(
        default_factory=list, description="Pairs of overlapping text labels"
    )
    suggestion: Optional[str] = Field(
        default=None, description="Suggestion for fixing overlaps if found"
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


__all__ = [
    "DiagramType",
    "DiagramAnalysis",
    "DiagramConfig",
    "OverlapCheckResult",
    "DiagramCandidate",
    "DiagramResult",
]
