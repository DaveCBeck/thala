"""Pydantic schemas for LLM structured output in illustrate workflow."""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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
    diagram_subtype: (
        Literal[
            "flowchart",
            "sequence",
            "concept_map",
            "network_graph",
            "hierarchy",
            "dependency_tree",
            "custom_artistic",
        ]
        | None
    ) = Field(
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
