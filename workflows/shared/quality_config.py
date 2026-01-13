"""Unified quality tier definitions for all workflows."""

from typing import Literal

# The single quality tier type used by all workflows
QualityTier = Literal["test", "quick", "standard", "comprehensive", "high_quality"]

QUALITY_TIER_DESCRIPTIONS = {
    "test": "Minimal processing for testing (~1 min)",
    "quick": "Fast results with limited depth (~5 min)",
    "standard": "Balanced quality and speed (~15 min)",
    "comprehensive": "Thorough processing (~30 min)",
    "high_quality": "Maximum depth and quality (45+ min)",
}
