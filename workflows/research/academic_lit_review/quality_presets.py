"""Quality tier configuration for academic literature review workflow."""

import os

from typing_extensions import TypedDict

# Read batch API preference from environment (default: True if not set)
_USE_BATCH_API = os.getenv("THALA_PREFER_BATCH_API", "1") == "1"


class QualitySettings(TypedDict):
    """Configuration for a quality tier."""

    max_stages: int  # Maximum diffusion stages
    max_papers: int  # Maximum papers to process
    target_word_count: int  # Target length of final review
    min_citations_filter: int  # Minimum citations for discovery
    saturation_threshold: float  # Coverage delta threshold
    use_batch_api: bool  # Use Anthropic Batch API
    supervision_loops: str  # Which supervision loops to run: "none", "one", "two", "three", "four", "all"
    recency_years: int  # Years to consider "recent" (default: 3)
    recency_quota: float  # Target fraction of recent papers (default: 0.25)


QUALITY_PRESETS: dict[str, QualitySettings] = {
    "test": QualitySettings(
        max_stages=1,
        max_papers=5,
        target_word_count=2000,
        min_citations_filter=0,
        saturation_threshold=0.5,
        use_batch_api=_USE_BATCH_API,
        supervision_loops="all",  # All loops with minimal iterations
        recency_years=3,
        recency_quota=0.0,  # Skip recency quota for test tier
    ),
    "quick": QualitySettings(
        max_stages=2,
        max_papers=50,
        target_word_count=8000,
        min_citations_filter=5,
        saturation_threshold=0.15,
        use_batch_api=_USE_BATCH_API,
        supervision_loops="all",  # All loops with minimal iterations
        recency_years=3,
        recency_quota=0.25,
    ),
    "standard": QualitySettings(
        max_stages=3,
        max_papers=100,
        target_word_count=12000,
        min_citations_filter=10,
        saturation_threshold=0.12,
        use_batch_api=_USE_BATCH_API,
        supervision_loops="all",  # Full multi-loop supervision
        recency_years=3,
        recency_quota=0.25,
    ),
    "comprehensive": QualitySettings(
        max_stages=4,
        max_papers=200,
        target_word_count=17500,
        min_citations_filter=10,
        saturation_threshold=0.10,
        use_batch_api=_USE_BATCH_API,
        supervision_loops="all",  # Full multi-loop supervision
        recency_years=3,
        recency_quota=0.25,
    ),
    "high_quality": QualitySettings(
        max_stages=5,
        max_papers=300,
        target_word_count=25000,
        min_citations_filter=10,
        saturation_threshold=0.10,
        use_batch_api=_USE_BATCH_API,
        supervision_loops="all",  # Full multi-loop supervision
        recency_years=3,
        recency_quota=0.25,
    ),
}
