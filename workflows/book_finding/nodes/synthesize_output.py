"""
Output synthesis node.

Creates the final markdown document with all book recommendations
organized by category.
"""

import logging
from datetime import datetime
from typing import Any

from workflows.shared.language import LanguageConfig, get_translated_prompt
from workflows.book_finding.state import BookResult, BookRecommendation

logger = logging.getLogger(__name__)

# Section headers for translation
SECTION_HEADERS_EN = {
    "book_recommendations": "Book Recommendations",
    "analogous_domain": "Analogous Domain",
    "analogous_subtitle": "Books exploring similar themes from different fields",
    "inspiring_action": "Inspiring Action",
    "inspiring_subtitle": "Books that inspire change and action",
    "expressive_fiction": "Expressive Fiction",
    "expressive_subtitle": "Fiction expressing the experience and potential of the theme",
    "additional_recommendations": "Additional recommendations (not processed)",
    "no_recommendations": "No recommendations in this category.",
    "summary": "Summary",
    "total_recommendations": "Total recommendations",
    "books_processed": "Books processed",
    "books_not_available": "Books not available/processed",
    "generated": "Generated",
}


async def _get_translated_headers(language_config: LanguageConfig | None) -> dict[str, str]:
    """Get translated section headers if needed."""
    if not language_config or language_config["code"] == "en":
        return SECTION_HEADERS_EN

    headers = {}
    for key, en_text in SECTION_HEADERS_EN.items():
        translated = await get_translated_prompt(
            en_text,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name=f"book_finding_header_{key}",
        )
        headers[key] = translated

    return headers


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
    language_config = state.get("language_config")

    # Get translated headers
    headers = await _get_translated_headers(language_config)

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
        f"# {headers['book_recommendations']}: {theme}",
        f"*{headers['generated']}: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
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
                lines.append(f"**{headers['additional_recommendations']}:**")
                lines.append("")
            for rec in unprocessed_in_section:
                lines.append(_format_recommendation_only(rec))
            lines.append("")

        if not processed_in_section and not unprocessed_in_section:
            lines.append(f"*{headers['no_recommendations']}*")
            lines.append("")

        return lines

    # Build each section
    sections.extend(build_section(
        headers["analogous_domain"],
        headers["analogous_subtitle"],
        analogous_recs,
    ))

    sections.extend(build_section(
        headers["inspiring_action"],
        headers["inspiring_subtitle"],
        inspiring_recs,
    ))

    sections.extend(build_section(
        headers["expressive_fiction"],
        headers["expressive_subtitle"],
        expressive_recs,
    ))

    # Add summary stats
    total_recs = len(analogous_recs) + len(inspiring_recs) + len(expressive_recs)
    total_processed = len(processed_books)
    failed = state.get("processing_failed", [])

    sections.extend([
        "---",
        "",
        f"## {headers['summary']}",
        f"- **{headers['total_recommendations']}:** {total_recs}",
        f"- **{headers['books_processed']}:** {total_processed}",
        f"- **{headers['books_not_available']}:** {len(failed)}",
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
