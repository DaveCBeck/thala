"""Pydantic models for structured clustering outputs."""

from pydantic import BaseModel, Field


class LLMThemeOutput(BaseModel):
    """Pydantic model for a single theme from LLM clustering."""

    name: str = Field(description="Clear, descriptive theme name suitable as a section heading")
    description: str = Field(description="2-3 sentence description of the theme")
    paper_dois: list[str] = Field(description="DOIs of papers belonging to this theme")
    sub_themes: list[str] = Field(default_factory=list, description="Sub-themes if the cluster is broad")
    relationships: list[str] = Field(default_factory=list, description="How this theme relates to other themes")


class LLMTopicSchemaOutput(BaseModel):
    """Pydantic model for LLM semantic clustering output."""

    themes: list[LLMThemeOutput] = Field(description="List of identified themes")
    reasoning: str = Field(description="Explanation of the clustering rationale")


class ClusterAnalysisOutput(BaseModel):
    """Pydantic model for deep analysis of a single cluster."""

    narrative_summary: str = Field(description="2-3 paragraph summary of the theme")
    timeline: list[str] = Field(default_factory=list, description="Key developments chronologically")
    key_debates: list[str] = Field(default_factory=list, description="Main debates and positions")
    methodologies: list[str] = Field(default_factory=list, description="Common methodological approaches")
    outstanding_questions: list[str] = Field(default_factory=list, description="Open research questions")


class ThematicClusterOutput(BaseModel):
    """Pydantic model for a synthesized thematic cluster."""

    cluster_id: int = Field(description="Unique cluster ID")
    label: str = Field(description="Final theme name")
    description: str = Field(description="What this cluster covers")
    paper_dois: list[str] = Field(description="DOIs of papers in this cluster")
    key_papers: list[str] = Field(default_factory=list, description="Most central papers")
    sub_themes: list[str] = Field(default_factory=list, description="Finer-grained topics")
    conflicts: list[str] = Field(default_factory=list, description="Contradictory findings")
    gaps: list[str] = Field(default_factory=list, description="Under-researched areas")
    source: str = Field(default="merged", description="Origin: bertopic, llm, or merged")


class OpusSynthesisOutput(BaseModel):
    """Pydantic model for Opus cluster synthesis output."""

    reasoning: str = Field(description="Explanation of synthesis decisions")
    final_clusters: list[ThematicClusterOutput] = Field(description="Synthesized thematic clusters")
