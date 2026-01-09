"""Deep research graph configuration."""

from typing_extensions import TypedDict


MAX_CONCURRENT_RESEARCHERS = 3

RECURSION_LIMITS = {
    "quick": 50,
    "standard": 100,
    "comprehensive": 200,
}


class WebResearchQualitySettings(TypedDict):
    """Configuration for a web research quality tier."""

    max_iterations: int
    recursion_limit: int
    description: str


# Quality presets for web research workflow
# Exported as QUALITY_PRESETS for consistency with other workflows
QUALITY_PRESETS: dict[str, WebResearchQualitySettings] = {
    "quick": WebResearchQualitySettings(
        max_iterations=2,
        recursion_limit=50,
        description="Fast research with 2 iterations (~5 min)",
    ),
    "standard": WebResearchQualitySettings(
        max_iterations=4,
        recursion_limit=100,
        description="Balanced research with 4 iterations (~15 min)",
    ),
    "comprehensive": WebResearchQualitySettings(
        max_iterations=8,
        recursion_limit=200,
        description="Exhaustive research with 8 iterations (30+ min)",
    ),
}
