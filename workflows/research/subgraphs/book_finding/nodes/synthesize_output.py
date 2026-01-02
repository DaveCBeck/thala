"""
Output synthesis node.

Creates the final markdown document with all book recommendations
organized by category.
"""

import logging
from datetime import datetime
from typing import Any

from workflows.research.subgraphs.book_finding.state import BookResult, BookRecommendation

logger = logging.getLogger(__name__)


def _find_recommendation_for_book(
    book: BookResult,
    recommendations: list[BookRecommendation],
) -> BookRecommendation | None:
    """Find the recommendation that matches this book."""
    matched_title = book.get("matched_recommendation", "")
    for rec in recommendations:
        if rec["title"] == matched_title:
            return rec
    return None


def _format_book_entry(
    book: BookResult,
    recommendation: BookRecommendation | None,
) -> str:
    """Format a single book entry for markdown output."""
    lines = [
        f"### {book['title']}",
        f"**Authors:** {book['authors']}",
    ]

    if recommendation:
        lines.append(f"**Why this book:** {recommendation['explanation']}")
    else:
        lines.append(f"**Matched recommendation:** {book.get('matched_recommendation', 'Unknown')}")

    lines.append("")

    if book.get("content_summary"):
        lines.append(f"**Summary:** {book['content_summary']}")
        lines.append("")

    return "\n".join(lines)


def _format_recommendation_only(rec: BookRecommendation) -> str:
    """Format a recommendation that wasn't found/processed."""
    author_str = f" by {rec['author']}" if rec.get("author") else ""
    return f"- **{rec['title']}**{author_str}: {rec['explanation']}"


async def synthesize_output(state: dict) -> dict[str, Any]:
    """Synthesize final markdown output.

    Creates a structured markdown document with:
    - Theme header
    - Three sections (analogous, inspiring, expressive)
    - Each section contains processed books with summaries
    - Unprocessed recommendations listed at end of each section
    """
    theme = state.get("input", {}).get("theme", "Unknown Theme")

    # Get all recommendations by category
    analogous_recs = state.get("analogous_recommendations", [])
    inspiring_recs = state.get("inspiring_recommendations", [])
    expressive_recs = state.get("expressive_recommendations", [])

    # Get processed books
    processed_books = state.get("processed_books", [])

    # Build a lookup of matched recommendation titles -> processed books
    processed_by_rec_title: dict[str, BookResult] = {}
    for book in processed_books:
        matched = book.get("matched_recommendation", "")
        if matched:
            processed_by_rec_title[matched] = book

    # Start building markdown
    sections = [
        f"# Book Recommendations: {theme}",
        f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "---",
        "",
    ]

    # Helper to build a category section
    def build_section(
        title: str,
        subtitle: str,
        recommendations: list[BookRecommendation],
    ) -> list[str]:
        lines = [
            f"## {title}",
            f"*{subtitle}*",
            "",
        ]

        processed_in_section = []
        unprocessed_in_section = []

        for rec in recommendations:
            if rec["title"] in processed_by_rec_title:
                processed_in_section.append(
                    (rec, processed_by_rec_title[rec["title"]])
                )
            else:
                unprocessed_in_section.append(rec)

        # Add processed books first (with summaries)
        for rec, book in processed_in_section:
            lines.append(_format_book_entry(book, rec))

        # Add unprocessed recommendations
        if unprocessed_in_section:
            if processed_in_section:
                lines.append("**Additional recommendations (not processed):**")
                lines.append("")
            for rec in unprocessed_in_section:
                lines.append(_format_recommendation_only(rec))
            lines.append("")

        if not processed_in_section and not unprocessed_in_section:
            lines.append("*No recommendations in this category.*")
            lines.append("")

        return lines

    # Build each section
    sections.extend(build_section(
        "Analogous Domain",
        "Books exploring similar themes from different fields",
        analogous_recs,
    ))

    sections.extend(build_section(
        "Inspiring Action",
        "Books that inspire change and action",
        inspiring_recs,
    ))

    sections.extend(build_section(
        "Expressive Fiction",
        "Fiction expressing the experience and potential of the theme",
        expressive_recs,
    ))

    # Add summary stats
    total_recs = len(analogous_recs) + len(inspiring_recs) + len(expressive_recs)
    total_processed = len(processed_books)
    failed = state.get("processing_failed", [])

    sections.extend([
        "---",
        "",
        "## Summary",
        f"- **Total recommendations:** {total_recs}",
        f"- **Books processed:** {total_processed}",
        f"- **Books not available/processed:** {len(failed)}",
    ])

    final_markdown = "\n".join(sections)

    logger.info(
        f"Synthesized markdown: {total_recs} recommendations, "
        f"{total_processed} processed"
    )

    return {
        "final_markdown": final_markdown,
        "completed_at": datetime.utcnow(),
        "current_phase": "completed",
    }
