"""
Book finding workflow.

A standalone workflow for discovering books related to a theme across
three categories:
1. Analogous domain - Books exploring theme from different fields
2. Inspiring action - Books that inspire change and action
3. Expressive fiction - Fiction capturing the theme's essence

Usage:
    from workflows.research.book_finding import book_finding

    result = await book_finding(
        theme="organizational resilience",
        brief="Focus on practical approaches",
    )
    print(result["final_markdown"])
"""

from workflows.research.book_finding.state import (
    BookFindingInput,
    BookFindingState,
    BookRecommendation,
    BookResult,
    BookFindingQualitySettings,
    QUALITY_PRESETS,
    BOOK_QUALITY_PRESETS,  # Backwards compat alias
)
from workflows.research.book_finding.graph import (
    book_finding_graph,
    book_finding,
)

__all__ = [
    # State types
    "BookFindingInput",
    "BookFindingState",
    "BookRecommendation",
    "BookResult",
    # Quality settings
    "BookFindingQualitySettings",
    "QUALITY_PRESETS",
    "BOOK_QUALITY_PRESETS",
    # Graph and API
    "book_finding_graph",
    "book_finding",
]
