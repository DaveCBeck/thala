"""Pydantic models for structured LLM outputs."""

from typing import Literal

from pydantic import BaseModel, Field


class EssayEvaluation(BaseModel):
    """Evaluation of a single essay by the choosing agent."""

    primary_strength: str = Field(description="The essay's main strength")
    primary_weakness: str = Field(description="The essay's main weakness")
    hook_strength: int = Field(ge=1, le=5, description="1-5 rating for opening hook")
    structural_momentum: int = Field(
        ge=1, le=5, description="1-5 rating for forward pull"
    )
    technical_payoff: int = Field(
        ge=1, le=5, description="1-5 rating for insight delivery"
    )
    tonal_calibration: int = Field(
        ge=1, le=5, description="1-5 rating for voice-material fit"
    )
    honest_complexity: int = Field(
        ge=1, le=5, description="1-5 rating for uncertainty acknowledgment"
    )
    subject_fit: int = Field(
        ge=1, le=5, description="1-5 rating for angle-material match"
    )


class ChoosingAgentOutput(BaseModel):
    """Structured output from the choosing agent."""

    selected: Literal["puzzle", "finding", "contrarian"] = Field(
        description="The winning essay angle"
    )

    evaluations: dict[str, EssayEvaluation] = Field(
        description="Evaluations keyed by angle: puzzle, finding, contrarian"
    )

    deciding_factors: str = Field(
        description="2-3 sentences explaining what made the winner stand out, "
        "with specific references to the text"
    )

    close_call: bool = Field(
        default=False,
        description="True if two essays were genuinely close in quality",
    )

    close_call_explanation: str = Field(
        default="",
        description="If close_call is True, explain what differentiates them",
    )
