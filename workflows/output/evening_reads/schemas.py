"""Pydantic models for structured LLM outputs."""

from typing import Literal

from pydantic import BaseModel, Field


class DeepDiveTopicPlan(BaseModel):
    """Plan for a single deep-dive article."""

    id: Literal["deep_dive_1", "deep_dive_2", "deep_dive_3"] = Field(
        description="Identifier for this deep-dive"
    )
    title: str = Field(
        description="Evocative, specific title (5-10 words). Should intrigue without being clickbait. "
        "Examples: 'The drilling ship and the smoking gun', 'When oceans turn toxic'"
    )
    theme: str = Field(
        description="2-3 sentence description of what this deep-dive covers and why it's distinct"
    )
    structural_approach: Literal["puzzle", "finding", "contrarian"] = Field(
        description="Narrative approach that best fits this topic's content. "
        "'puzzle': Opens with mystery/anomaly, unfolds as investigation. Best for mysteries, unexpected findings. "
        "'finding': Leads with striking quantitative result, explores implications. Best for data-driven topics. "
        "'contrarian': Steelmans assumption then complicates it. Best for overturning conventional wisdom."
    )
    anchor_keys: list[str] = Field(
        description="2-3 Zotero citation keys that anchor this deep-dive. "
        "These are the primary sources whose content will be fetched."
    )
    relevant_sections: list[str] = Field(
        description="Section headers or topics from the literature review that this deep-dive should draw from"
    )
    distinctiveness_rationale: str = Field(
        description="1-2 sentences explaining why this topic doesn't overlap with the other two deep-dives"
    )


class PlanningOutput(BaseModel):
    """Structured output from the planning node."""

    deep_dives: list[DeepDiveTopicPlan] = Field(
        description="Exactly 3 deep-dive plans, each covering a distinct aspect",
        min_length=3,
        max_length=3,
    )

    overview_scope: str = Field(
        description="3-4 sentences describing what the overview article should cover. "
        "The overview synthesizes the big picture and references the deep-dives without duplicating them."
    )

    series_coherence: str = Field(
        description="2-3 sentences explaining how these 4 pieces work together as a series"
    )

    def get_all_anchor_keys(self) -> set[str]:
        """Get all unique anchor keys across all deep-dives."""
        keys = set()
        for dd in self.deep_dives:
            keys.update(dd.anchor_keys)
        return keys

    def get_anchor_keys_for(self, deep_dive_id: str) -> list[str]:
        """Get anchor keys for a specific deep-dive."""
        for dd in self.deep_dives:
            if dd.id == deep_dive_id:
                return dd.anchor_keys
        return []
