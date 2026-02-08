"""Pydantic schemas for LLM structured output in illustrate workflow."""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Shared literal type for diagram subtypes
DiagramSubtype = Literal[
    "flowchart",
    "sequence",
    "concept_map",
    "network_graph",
    "hierarchy",
    "dependency_tree",
    "custom_artistic",
]


def _parse_json_string_list(v: Any) -> list:
    """Handle LLM returning JSON string instead of list."""
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        if v.strip():
            return [v]
        return []
    return v if v is not None else []


# ---------------------------------------------------------------------------
# Two-pass planning schemas (creative_direction + plan_briefs)
# ---------------------------------------------------------------------------


class VisualIdentity(BaseModel):
    """Consistent visual style across all images in one article."""

    primary_style: str = Field(
        description="e.g., 'editorial watercolor illustration'",
    )
    color_palette: list[str] = Field(
        description="3-5 descriptive color names, e.g., ['warm amber', 'deep teal', 'ivory']",
    )
    mood: str = Field(
        description="e.g., 'contemplative, intellectual, accessible'",
    )
    lighting: str = Field(
        description="e.g., 'soft diffused natural light'",
    )
    avoid: list[str] = Field(
        description="e.g., ['photorealistic faces', 'neon colors']",
    )
    palette_hex: list[str] = Field(
        default_factory=list,
        description="Hex codes resolved from color_palette, for diagram injection.",
    )

    @field_validator("color_palette", "avoid", mode="before")
    @classmethod
    def parse_json_string_list(cls, v: Any) -> list:
        return _parse_json_string_list(v)

    @field_validator("palette_hex", mode="before")
    @classmethod
    def parse_palette_hex_json(cls, v: Any) -> list:
        return _parse_json_string_list(v)


class ImageOpportunity(BaseModel):
    """A candidate location for an image, identified in Pass 1."""

    location_id: str = Field(
        description="Unique identifier like 'header', 'section_1', 'section_2'",
    )
    insertion_after_header: str = Field(
        description="The exact header text after which to insert the image",
    )
    purpose: Literal["header", "illustration", "diagram"] = Field(
        description="Role of this image in the document",
    )
    suggested_type: Literal["generated", "public_domain", "diagram"] = Field(
        description="Suggested generation method",
    )
    strength: Literal["strong", "stretch"] = Field(
        description="'strong' = clearly benefits from an image, 'stretch' = nice-to-have",
    )
    rationale: str = Field(
        description="Brief rationale for why this location benefits from an image",
    )
    diagram_subtype: DiagramSubtype | None = Field(
        default=None,
        description="For diagrams: subtype determines rendering engine.",
    )


class CreativeDirectionResult(BaseModel):
    """Full output of Pass 1: creative_direction node."""

    document_title: str = Field(
        description="Extracted or confirmed document title",
    )
    visual_identity: VisualIdentity
    image_opportunities: list[ImageOpportunity] = Field(
        description="Candidate locations for images (target + 2 extras)",
    )
    editorial_notes: str = Field(
        description="Tone, pacing, variety guidance",
    )

    @field_validator("image_opportunities", mode="before")
    @classmethod
    def parse_opportunities_json(cls, v: Any) -> list:
        return _parse_json_string_list(v)


class CandidateBrief(BaseModel):
    """A single brief for one candidate at a location, from Pass 2."""

    location_id: str = Field(
        description="References an ImageOpportunity.location_id",
    )
    candidate_index: int = Field(
        ge=1,
        le=2,
        description="1 or 2 (up to 2 per location)",
    )
    image_type: Literal["generated", "public_domain", "diagram"] = Field(
        description="Which generation method to use",
    )
    brief: str = Field(
        description="Full brief text for image generation or search",
    )
    relationship_to_text: Literal["literal", "metaphorical", "explanatory", "evocative"] = Field(
        description="How this image relates to the surrounding text",
    )
    visual_identity_references: str = Field(
        description="How this brief references the palette/mood/style",
    )
    # Stock photo fields
    literal_queries: list[str] = Field(
        default_factory=list,
        description="For public_domain: direct subject matter search terms.",
    )
    conceptual_queries: list[str] = Field(
        default_factory=list,
        description="For public_domain: mood/metaphor search terms.",
    )
    query_strategy: Literal["literal", "conceptual", "both"] | None = Field(
        default=None,
        description="Which query type best matches the brief's intent.",
    )
    # Diagram fields
    diagram_subtype: DiagramSubtype | None = Field(
        default=None,
        description="For diagrams: subtype determines rendering engine.",
    )

    @field_validator("literal_queries", "conceptual_queries", mode="before")
    @classmethod
    def parse_queries_json(cls, v: Any) -> list:
        return _parse_json_string_list(v)


class PlanBriefsResult(BaseModel):
    """Full output of Pass 2: plan_briefs node."""

    candidate_briefs: list[CandidateBrief] = Field(
        description="Up to 2 candidate briefs per selected opportunity",
    )
    brief_strategy_notes: str = Field(
        description="Cross-location variety decisions",
    )

    @field_validator("candidate_briefs", mode="before")
    @classmethod
    def parse_briefs_json(cls, v: Any) -> list:
        return _parse_json_string_list(v)


class ImageLocationPlan(BaseModel):
    """Plan for a single image location in the document."""

    location_id: str = Field(
        description="Unique identifier like 'header', 'section_1', 'section_2'",
    )
    insertion_after_header: str = Field(
        description="The exact header text after which to insert the image",
    )
    purpose: Literal["header", "illustration", "diagram"] = Field(
        description="Role of this image in the document",
    )
    image_type: Literal["generated", "public_domain", "diagram"] = Field(
        description="Which generation method to use",
    )
    type_rationale: str | None = Field(
        default=None,
        description="Brief explanation of why this image type was chosen",
    )
    brief: str = Field(
        description=(
            "Detailed brief for image generation. For generated images, this is "
            "the Imagen prompt. For public domain, these are selection criteria. "
            "For diagrams, these are structural requirements. Can include "
            "extensive document context when needed."
        ),
    )
    search_query: str | None = Field(
        default=None,
        description="For public domain images, the search query to use",
    )
    literal_queries: list[str] = Field(
        default_factory=list,
        description="Search terms for literal depictions of the topic.",
    )
    conceptual_queries: list[str] = Field(
        default_factory=list,
        description="Search terms that EVOKE the feeling/mood, NOT literal depictions.",
    )
    query_strategy: Literal["literal", "conceptual", "both"] | None = Field(
        default=None,
        description="Which query type best matches the brief's intent.",
    )
    diagram_subtype: DiagramSubtype | None = Field(
        default=None,
        description="For diagrams: subtype determines rendering engine. "
        "flowchart/sequence/concept_map → Mermaid, "
        "network_graph/hierarchy/dependency_tree → Graphviz, "
        "custom_artistic → raw SVG.",
    )


class DocumentAnalysis(BaseModel):
    """Full analysis of where images should go in document."""

    document_title: str = Field(
        description="Extracted or confirmed document title",
    )
    header_image: ImageLocationPlan = Field(
        description="Plan for the header/lead image",
    )
    additional_images: list[ImageLocationPlan] = Field(
        description="Additional image locations (1-2 typically)",
        min_length=0,
        max_length=5,
    )
    analysis_notes: str = Field(
        description="Brief notes on the visual strategy for this document",
    )


class VisionReviewResult(BaseModel):
    """Result of vision review of a generated image."""

    fits_context: bool = Field(
        description="Whether the image fits the document context",
    )
    has_substantive_errors: bool = Field(
        description="True if there are factual/substantive errors requiring regeneration",
    )
    has_minor_issues: bool = Field(
        description="True if there are minor issues (log warning but accept)",
    )
    issues: list[str] = Field(
        description="List of identified issues",
        default_factory=list,
    )
    recommendation: Literal["accept", "accept_with_warning", "retry", "fail"] = Field(
        description="What to do with this image",
    )
    improved_brief: str | None = Field(
        default=None,
        description="If retry recommended, an improved brief/prompt",
    )

    @field_validator("issues", mode="before")
    @classmethod
    def parse_issues_json_string(cls, v: Any) -> list:
        return _parse_json_string_list(v)


class ImageCompareResult(BaseModel):
    """Result of comparing multiple image candidates."""

    selected_candidate: int = Field(
        ge=1,
        le=5,
        description="Which candidate is best (1-indexed)",
    )
    reasoning: str = Field(
        description="Brief explanation of why this candidate was selected",
    )
    issues_with_selected: list[str] = Field(
        default_factory=list,
        description="Any remaining issues with the selected image (for logging)",
    )

    @field_validator("issues_with_selected", mode="before")
    @classmethod
    def parse_issues_json_string(cls, v: Any) -> list:
        return _parse_json_string_list(v)


class HeaderAppositenessResult(BaseModel):
    """Result of evaluating whether a public domain image is 'apposite' for header."""

    is_apposite: bool = Field(
        description="Whether this public domain image is particularly fitting for the header",
    )
    reasoning: str = Field(
        description="Explanation of why/why not the image is apposite",
    )
    quality_score: int = Field(
        ge=1,
        le=5,
        description="Quality score 1-5: 1=poor fit, 5=excellent fit",
    )
    suggested_search_query: str | None = Field(
        default=None,
        description="If not apposite, suggest a better search query that would find a more fitting image",
    )
