"""Quality tier configuration for agent-team web research."""

from typing_extensions import TypedDict

from workflows.shared.quality_config import QualityTier


class TeamQualitySettings(TypedDict):
    max_iterations: int
    max_researcher_turns: int
    description: str


QUALITY_PRESETS: dict[QualityTier, TeamQualitySettings] = {
    "test": TeamQualitySettings(
        max_iterations=1,
        max_researcher_turns=10,
        description="Smoke test (~2 min)",
    ),
    "quick": TeamQualitySettings(
        max_iterations=3,
        max_researcher_turns=20,
        description="Fast research (~10 min)",
    ),
    "standard": TeamQualitySettings(
        max_iterations=5,
        max_researcher_turns=20,
        description="Balanced research (~20 min)",
    ),
    "comprehensive": TeamQualitySettings(
        max_iterations=7,
        max_researcher_turns=25,
        description="Thorough research (~40 min)",
    ),
    "high_quality": TeamQualitySettings(
        max_iterations=10,
        max_researcher_turns=30,
        description="Maximum depth (60+ min)",
    ),
}
