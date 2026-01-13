"""Supervisor tool schemas (Pydantic for tool binding)."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class ConductResearch(BaseModel):
    """Tool for delegating a research task to a specialized sub-agent."""

    research_topic: str = Field(
        description="The topic to research. Should be a single topic described in high detail (at least a paragraph)."
    )


class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""

    pass


class RefineDraftReport(BaseModel):
    """Tool for refining the draft report with new findings."""

    updates: str = Field(
        description="The updates to make to the draft report based on new findings."
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Remaining gaps that still need research.",
    )


class SupervisorDecision(BaseModel):
    """Supervisor's structured decision for the next research step.

    Used with structured output to ensure clean, parseable decisions
    without metadata contamination.
    """

    action: Literal["conduct_research", "refine_draft", "research_complete"] = Field(
        description="The next action to take in the research process."
    )

    # For conduct_research action
    research_questions: list[str] = Field(
        default_factory=list,
        description="1-3 specific research questions to investigate. Must be actual questions about the research topic - NOT analysis notes, metadata, or summaries of previous findings.",
        max_length=3,
    )

    # Researcher allocation (for conduct_research action)
    web_researchers: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Number of web researchers (1-3) to run in parallel.",
    )

    # For refine_draft action
    draft_updates: Optional[str] = Field(
        default=None,
        description="Content to add or update in the draft report based on new findings.",
    )
    remaining_gaps: list[str] = Field(
        default_factory=list,
        description="Research gaps that still need investigation.",
    )

    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why this action was chosen.",
    )
