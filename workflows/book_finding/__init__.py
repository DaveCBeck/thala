"""
Book finding workflow.

A standalone workflow for discovering books related to a theme across
three categories:
1. Analogous domain - Books exploring theme from different fields
2. Inspiring action - Books that inspire change and action
3. Expressive fiction - Fiction capturing the theme's essence

Usage:
    from workflows.book_finding import book_finding

    result = await book_finding(
        theme="organizational resilience",
        brief="Focus on practical approaches",
    )
    print(result["final_markdown"])
"""

from workflows.book_finding.state import (
    BookFindingInput,
    BookFindingState,
    BookRecommendation,
    BookResult,
)
from workflows.book_finding.graph import (
    book_finding_graph,
    book_finding,
)

__all__ = [
    # State types
    "BookFindingInput",
    "BookFindingState",
    "BookRecommendation",
    "BookResult",
    # Graph and API
    "book_finding_graph",
    "book_finding",
]
