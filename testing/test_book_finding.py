#!/usr/bin/env python3
"""
Test script for the book finding workflow.

Runs the book finding workflow to discover relevant books across three categories:
1. Analogous domain - Books exploring theme from different fields
2. Inspiring action - Books that inspire change and action
3. Expressive fiction - Fiction capturing the theme's essence

Usage:
    python test_book_finding.py "your theme"
    python test_book_finding.py "organizational resilience" quick
    python test_book_finding.py "creative leadership" standard --language es
    python test_book_finding.py  # Uses default theme

Valid quality levels: quick, standard, comprehensive (default: quick)
Valid languages: en, es, zh, ja, de, fr, pt, ko, ru, ar, it, nl, pl, tr, vi, th, id, hi, bn, sv, no, da, fi, cs, el, he, uk, ro, hu (default: en)

Environment:
    Set THALA_MODE=dev in .env to enable LangSmith tracing
"""

import asyncio
import os
from datetime import datetime

# Enable dev mode for LangSmith tracing before any imports
os.environ["THALA_MODE"] = "dev"

from testing.utils import (
    setup_logging,
    get_output_dir,
    save_json_result,
    save_markdown_report,
    print_section_header,
    safe_preview,
    print_timing,
    print_errors,
    print_quality_analysis,
    BaseQualityAnalyzer,
    QualityMetrics,
    create_test_parser,
    add_quality_argument,
    add_language_argument,
)

# Setup logging
logger = setup_logging("book_finding")

# Output directory for results
OUTPUT_DIR = get_output_dir()

VALID_QUALITIES = ["quick", "standard", "comprehensive"]
DEFAULT_QUALITY = "quick"


class BookFindingQualityAnalyzer(BaseQualityAnalyzer):
    """Quality analyzer for book finding workflow results."""

    output_field = "final_report"
    output_field_alt = "final_markdown"
    min_word_count = 200
    min_source_count = 3

    def _analyze_workflow_specific(self, metrics: QualityMetrics) -> None:
        """Analyze book-specific metrics."""
        analogous = self.result.get("analogous_recommendations", [])
        inspiring = self.result.get("inspiring_recommendations", [])
        expressive = self.result.get("expressive_recommendations", [])

        metrics.workflow_specific["analogous_count"] = len(analogous)
        metrics.workflow_specific["inspiring_count"] = len(inspiring)
        metrics.workflow_specific["expressive_count"] = len(expressive)
        metrics.workflow_specific["total_recommendations"] = len(analogous) + len(inspiring) + len(expressive)

        # Processing metrics
        processed = self.result.get("processed_books", [])
        failed = self.result.get("processing_failed", [])
        search_results = self.result.get("search_results", [])

        metrics.workflow_specific["books_found"] = len(search_results)
        metrics.workflow_specific["books_processed"] = len(processed)
        metrics.workflow_specific["books_failed"] = len(failed)

        total_recommended = metrics.workflow_specific["total_recommendations"]
        if total_recommended > 0 and search_results:
            metrics.workflow_specific["search_success_rate"] = len(search_results) / total_recommended

        if search_results:
            metrics.workflow_specific["processing_success_rate"] = len(processed) / len(search_results)

    def _identify_issues(self, metrics: QualityMetrics) -> None:
        """Identify book-finding specific issues."""
        super()._identify_issues(metrics)

        if metrics.workflow_specific.get("analogous_count", 0) == 0:
            metrics.issues.append("No analogous domain recommendations generated")
        if metrics.workflow_specific.get("inspiring_count", 0) == 0:
            metrics.issues.append("No inspiring action recommendations generated")
        if metrics.workflow_specific.get("expressive_count", 0) == 0:
            metrics.issues.append("No expressive fiction recommendations generated")

        search_rate = metrics.workflow_specific.get("search_success_rate", 1)
        if search_rate < 0.5:
            metrics.issues.append(f"Low search success rate ({search_rate:.0%})")

        process_rate = metrics.workflow_specific.get("processing_success_rate", 1)
        if process_rate < 0.5:
            metrics.issues.append(f"Low processing success rate ({process_rate:.0%})")

    def _generate_suggestions(self, metrics: QualityMetrics) -> None:
        """Generate book-finding specific suggestions."""
        if not metrics.issues:
            metrics.suggestions.append("Book finding completed successfully")
            return

        if "search success rate" in str(metrics.issues).lower():
            metrics.suggestions.append("Some recommended books may not be available - this is normal")
        if "processing success rate" in str(metrics.issues).lower():
            metrics.suggestions.append("PDF processing can fail for various reasons - summaries still generated from recommendations")


def print_result_summary(result: dict, theme: str) -> None:
    """Print a detailed summary of the book finding result."""
    print_section_header("BOOK FINDING RESULT")

    # Theme
    print(f"\nTheme: {theme}")

    # Timing
    print_timing(result.get("started_at"), result.get("completed_at"))

    # Recommendations
    analogous = result.get("analogous_recommendations", [])
    inspiring = result.get("inspiring_recommendations", [])
    expressive = result.get("expressive_recommendations", [])

    print(f"\n--- Recommendations ---")
    print(f"Analogous Domain: {len(analogous)} books")
    for book in analogous:
        title = book.get("title", "Unknown")
        author = book.get("author", "Unknown")
        print(f"  - {title} by {author}")

    print(f"\nInspiring Action: {len(inspiring)} books")
    for book in inspiring:
        title = book.get("title", "Unknown")
        author = book.get("author", "Unknown")
        print(f"  - {title} by {author}")

    print(f"\nExpressive Fiction: {len(expressive)} books")
    for book in expressive:
        title = book.get("title", "Unknown")
        author = book.get("author", "Unknown")
        print(f"  - {title} by {author}")

    # Processed books
    processed = result.get("processed_books", [])
    failed = result.get("processing_failed", [])
    search_results = result.get("search_results", [])

    print(f"\n--- Processing ---")
    print(f"Books found in search: {len(search_results)}")
    print(f"Books successfully processed: {len(processed)}")
    print(f"Books failed to process: {len(failed)}")

    if processed:
        print("\nProcessed books:")
        for book in processed[:5]:
            title = book.get("title", "Unknown")
            authors = book.get("authors", [])
            author_str = ", ".join(authors[:2]) if authors else "Unknown"
            print(f"  - {title} by {author_str}")
        if len(processed) > 5:
            print(f"  ... and {len(processed) - 5} more")

    if failed:
        print("\nFailed to process:")
        for title in failed[:5]:
            print(f"  - {title}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    # Final report preview
    final_report = result.get("final_report") or result.get("final_markdown", "")
    if final_report:
        print(f"\n--- Final Output ---")
        word_count = len(final_report.split())
        print(f"Length: {len(final_report)} chars ({word_count} words)")
        print(safe_preview(final_report, 1000))

    # Errors
    print_errors(result.get("errors", []))

    print("\n" + "=" * 80)


async def run_book_finding(
    theme: str,
    brief: str | None = None,
    quality: str = "quick",
    language: str = "en",
) -> dict:
    """Run the book finding workflow on a theme."""
    from workflows.book_finding import book_finding

    logger.info(f"Starting book finding for theme: {theme}")
    logger.info(f"Quality: {quality}")
    logger.info(f"Language: {language}")
    logger.info(f"LangSmith tracing: {os.environ.get('LANGSMITH_TRACING', 'false')}")

    result = await book_finding(
        theme=theme,
        brief=brief,
        quality=quality,
        language=language,
    )

    return result


def parse_args():
    """Parse command line arguments."""
    parser = create_test_parser(
        description="Run book finding workflow",
        default_topic="The intersection of technology and human creativity",
        topic_help="Theme to explore for book recommendations",
        epilog_examples="""
Examples:
  %(prog)s "organizational resilience"              # Quick book finding
  %(prog)s "creative leadership" standard           # Standard quality
  %(prog)s "digital transformation" comprehensive   # Comprehensive search
  %(prog)s "liderazgo creativo" quick --language es # Spanish output
        """
    )

    add_quality_argument(parser, choices=VALID_QUALITIES, default=DEFAULT_QUALITY)
    add_language_argument(parser)

    parser.add_argument(
        "--brief", "-b",
        type=str,
        default=None,
        help="Additional context to guide recommendations"
    )

    return parser.parse_args()


async def main():
    """Run book finding test."""
    args = parse_args()

    theme = args.topic  # Note: topic arg is used as theme
    quality = args.quality
    brief = args.brief
    language = args.language

    print_section_header("BOOK FINDING TEST")
    print(f"\nTheme: {theme}")
    print(f"Quality: {quality}")
    print(f"Language: {language}")
    if brief:
        print(f"Brief: {brief}")
    print(f"LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'thala-dev')}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        result = await run_book_finding(
            theme=theme,
            brief=brief,
            quality=quality,
            language=language,
        )

        # Print detailed result summary
        print_result_summary(result, theme)

        # Analyze quality
        analyzer = BookFindingQualityAnalyzer(result)
        metrics = analyzer.analyze()
        print_quality_analysis(metrics)

        # Save results
        result_file = save_json_result(result, "book_finding_result")
        logger.info(f"Full result saved to: {result_file}")

        analysis_file = save_json_result(metrics.to_dict(), "book_finding_analysis")
        logger.info(f"Analysis saved to: {analysis_file}")

        # Save final output as markdown
        final_report = result.get("final_report") or result.get("final_markdown")
        if final_report:
            report_file = save_markdown_report(
                final_report,
                "book_finding",
                title=f"Book Recommendations: {theme}",
                metadata={"quality": quality, "language": language},
            )
            logger.info(f"Report saved to: {report_file}")

        return result, metrics.to_dict()

    except Exception as e:
        logger.error(f"Book finding failed: {e}", exc_info=True)
        return {"error": str(e)}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
