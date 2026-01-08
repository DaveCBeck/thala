"""Supervised academic literature review workflow.

This workflow wraps the core academic_lit_review and adds multi-loop
supervision to enhance review quality through:

- Loop 1: Theoretical depth expansion
- Loop 2: Literature base expansion (missing perspectives)
- Loop 3: Structure and cohesion editing
- Loop 4: Section-level deep editing
- Loop 4.5: Cohesion check (may return to Loop 3)
- Loop 5: Fact and reference checking

Usage:
    from workflows.supervised_lit_review import supervised_lit_review

    result = await supervised_lit_review(
        topic="Your research topic",
        research_questions=["Question 1?", "Question 2?"],
        quality="high_quality",
        supervision_loops="all",  # or "none", "one", "two", etc.
    )
"""

from workflows.supervised_lit_review.api import supervised_lit_review

__all__ = ["supervised_lit_review"]
