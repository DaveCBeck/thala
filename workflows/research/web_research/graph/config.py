"""Deep research graph configuration."""

from typing_extensions import TypedDict

from workflows.shared.quality_config import QualityTier


MAX_CONCURRENT_RESEARCHERS = 3

RECURSION_LIMITS: dict[QualityTier, int] = {
    "test": 25,
    "quick": 50,
    "standard": 100,
    "comprehensive": 200,
    "high_quality": 300,
}


class WebResearchQualitySettings(TypedDict):
    """Configuration for a web research quality tier."""

    max_iterations: int
    recursion_limit: int
    description: str


# Quality presets for web research workflow
QUALITY_PRESETS: dict[QualityTier, WebResearchQualitySettings] = {
    "test": WebResearchQualitySettings(
        max_iterations=1,
        recursion_limit=25,
        description="Minimal testing with 1 iteration (~1 min)",
    ),
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
    "high_quality": WebResearchQualitySettings(
        max_iterations=12,
        recursion_limit=300,
        description="Maximum depth with 12 iterations (45+ min)",
    ),
}
