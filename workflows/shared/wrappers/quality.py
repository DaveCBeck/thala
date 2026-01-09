"""
Quality tier mapping for workflow orchestration.

Maps unified quality tiers (quick/standard/comprehensive) to
workflow-specific quality settings.
"""

from typing import Literal


# Unified quality tiers available to wrapper workflows
QualityTier = Literal["quick", "standard", "comprehensive"]

# Maps unified quality tier to each workflow's specific quality parameter
# Note: web_research uses "depth", others use "quality"
QUALITY_MAPPING: dict[str, dict[str, str]] = {
    "quick": {
        "web_research": "quick",
        "academic_lit_review": "quick",
        "book_finding": "quick",
        "supervised_lit_review": "quick",
    },
    "standard": {
        "web_research": "standard",
        "academic_lit_review": "standard",
        "book_finding": "standard",
        "supervised_lit_review": "standard",
    },
    "comprehensive": {
        "web_research": "comprehensive",
        "academic_lit_review": "high_quality",  # Uses 5-tier system
        "book_finding": "comprehensive",
        "supervised_lit_review": "high_quality",  # Uses 5-tier system
    },
}


def get_workflow_quality(unified_tier: str, workflow_key: str) -> str:
    """Get the workflow-specific quality setting for a unified tier.

    Args:
        unified_tier: The unified quality tier ("quick", "standard", "comprehensive")
        workflow_key: The workflow identifier (e.g., "web_research", "academic_lit_review")

    Returns:
        The workflow-specific quality parameter value

    Raises:
        KeyError: If unified_tier or workflow_key is not found
    """
    return QUALITY_MAPPING[unified_tier][workflow_key]


def get_quality_tiers() -> list[str]:
    """Get the list of available unified quality tiers.

    Returns:
        List of tier names: ["quick", "standard", "comprehensive"]
    """
    return list(QUALITY_MAPPING.keys())
