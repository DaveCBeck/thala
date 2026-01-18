"""Quality tier configuration for the fact-check workflow."""

from typing_extensions import TypedDict


class FactCheckQualitySettings(TypedDict):
    """Configuration for a fact-check quality tier."""

    # Verification settings
    verify_use_perplexity: bool  # Enable Perplexity fact-checking
    verify_confidence_threshold: float  # Min confidence to consider resolved
    fact_check_max_tool_calls: int  # Max tool calls per fact-check section
    reference_check_max_tool_calls: int  # Max tool calls per reference-check (base, +5 per citation)


FACT_CHECK_QUALITY_PRESETS: dict[str, FactCheckQualitySettings] = {
    "test": FactCheckQualitySettings(
        verify_use_perplexity=False,  # Skip for speed in test
        verify_confidence_threshold=0.7,
        fact_check_max_tool_calls=5,
        reference_check_max_tool_calls=3,
    ),
    "quick": FactCheckQualitySettings(
        verify_use_perplexity=True,
        verify_confidence_threshold=0.7,
        fact_check_max_tool_calls=10,
        reference_check_max_tool_calls=5,
    ),
    "standard": FactCheckQualitySettings(
        verify_use_perplexity=True,
        verify_confidence_threshold=0.75,
        fact_check_max_tool_calls=15,
        reference_check_max_tool_calls=8,
    ),
    "comprehensive": FactCheckQualitySettings(
        verify_use_perplexity=True,
        verify_confidence_threshold=0.8,
        fact_check_max_tool_calls=18,
        reference_check_max_tool_calls=10,
    ),
    "high_quality": FactCheckQualitySettings(
        verify_use_perplexity=True,
        verify_confidence_threshold=0.85,
        fact_check_max_tool_calls=20,
        reference_check_max_tool_calls=12,
    ),
}


def get_fact_check_quality_settings(quality: str) -> FactCheckQualitySettings:
    """Get quality settings for a tier.

    Args:
        quality: Quality tier name

    Returns:
        Quality settings dict
    """
    if quality not in FACT_CHECK_QUALITY_PRESETS:
        return FACT_CHECK_QUALITY_PRESETS["standard"]
    return FACT_CHECK_QUALITY_PRESETS[quality].copy()
