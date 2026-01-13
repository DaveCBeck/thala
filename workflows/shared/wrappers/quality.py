"""
Quality tier utilities for workflow orchestration.

All workflows now use the same unified 5-tier quality system:
test, quick, standard, comprehensive, high_quality

This module re-exports the shared QualityTier type and provides
utility functions for quality tier management.
"""

from workflows.shared.quality_config import QualityTier, QUALITY_TIER_DESCRIPTIONS

# Re-export for backward compatibility
__all__ = ["QualityTier", "QUALITY_TIER_DESCRIPTIONS", "get_quality_tiers"]


def get_quality_tiers() -> list[str]:
    """Get the list of available quality tiers.

    Returns:
        List of tier names: ["test", "quick", "standard", "comprehensive", "high_quality"]
    """
    return list(QUALITY_TIER_DESCRIPTIONS.keys())
