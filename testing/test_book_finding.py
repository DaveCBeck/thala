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

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable dev mode for LangSmith tracing before any imports
os.environ["THALA_MODE"] = "dev"

# Setup logging - both console and file
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Create a unique log file for each run
_log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = LOG_DIR / f"book_finding_{_log_timestamp}.log"

# Configure root logger for both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(LOG_FILE, mode='w'),  # File output
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {LOG_FILE}")

# Output directory for results
OUTPUT_DIR = Path(__file__).parent / "test_data"

VALID_QUALITIES = ["quick", "standard", "comprehensive"]
DEFAULT_QUALITY = "quick"


def print_result_summary(result: dict, theme: str) -> None:
    """Print a detailed summary of the book finding result."""
    print("\n" + "=" * 80)
    print("BOOK FINDING RESULT")
    print("=" * 80)

    # Theme
    print(f"\nTheme: {theme}")

    # Timing
    started = result.get("started_at")
    completed = result.get("completed_at")
    if started and completed:
        try:
            if isinstance(started, str):
                started = datetime.fromisoformat(started.replace("Z", "+00:00"))
            if isinstance(completed, str):
                completed = datetime.fromisoformat(completed.replace("Z", "+00:00"))
            if hasattr(started, 'replace') and started.tzinfo is not None:
                started = started.replace(tzinfo=None)
            if hasattr(completed, 'replace') and completed.tzinfo is not None:
                completed = completed.replace(tzinfo=None)
            duration = (completed - started).total_seconds()
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            print(f"Duration: {minutes}m {seconds}s ({duration:.1f}s total)")
        except Exception as e:
            print(f"Duration: (error calculating: {e})")

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

    # Final markdown preview
    final_markdown = result.get("final_markdown", "")
    if final_markdown:
        print(f"\n--- Final Output ---")
        word_count = len(final_markdown.split())
        print(f"Length: {len(final_markdown)} chars ({word_count} words)")
        # Show first 1000 chars
        preview = final_markdown[:1000]
        if len(final_markdown) > 1000:
            preview += "\n\n... [truncated] ..."
        print(preview)

    # Errors
    errors = result.get("errors", [])
    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for err in errors:
            phase = err.get("phase", "unknown")
            error = err.get("error", "unknown")
            print(f"  [{phase}]: {error}")

    print("\n" + "=" * 80)


def analyze_quality(result: dict) -> dict:
    """Analyze book finding quality and return metrics."""
    analysis = {
        "metrics": {},
        "issues": [],
        "suggestions": [],
    }

    # Recommendation metrics
    analogous = result.get("analogous_recommendations", [])
    inspiring = result.get("inspiring_recommendations", [])
    expressive = result.get("expressive_recommendations", [])

    analysis["metrics"]["analogous_count"] = len(analogous)
    analysis["metrics"]["inspiring_count"] = len(inspiring)
    analysis["metrics"]["expressive_count"] = len(expressive)
    analysis["metrics"]["total_recommendations"] = len(analogous) + len(inspiring) + len(expressive)

    if len(analogous) == 0:
        analysis["issues"].append("No analogous domain recommendations generated")
    if len(inspiring) == 0:
        analysis["issues"].append("No inspiring action recommendations generated")
    if len(expressive) == 0:
        analysis["issues"].append("No expressive fiction recommendations generated")

    # Processing metrics
    processed = result.get("processed_books", [])
    failed = result.get("processing_failed", [])
    search_results = result.get("search_results", [])

    analysis["metrics"]["books_found"] = len(search_results)
    analysis["metrics"]["books_processed"] = len(processed)
    analysis["metrics"]["books_failed"] = len(failed)

    total_recommended = analysis["metrics"]["total_recommendations"]
    if total_recommended > 0:
        search_rate = len(search_results) / total_recommended
        analysis["metrics"]["search_success_rate"] = search_rate
        if search_rate < 0.5:
            analysis["issues"].append(f"Low search success rate ({search_rate:.0%})")

    if len(search_results) > 0:
        process_rate = len(processed) / len(search_results)
        analysis["metrics"]["processing_success_rate"] = process_rate
        if process_rate < 0.5:
            analysis["issues"].append(f"Low processing success rate ({process_rate:.0%})")

    # Output quality
    final_markdown = result.get("final_markdown", "")
    if final_markdown and not final_markdown.startswith("# Book Finding Failed"):
        word_count = len(final_markdown.split())
        analysis["metrics"]["output_word_count"] = word_count
        analysis["metrics"]["completed"] = True
    else:
        analysis["metrics"]["completed"] = False
        analysis["issues"].append("Book finding failed or incomplete")

    # Errors
    errors = result.get("errors", [])
    analysis["metrics"]["error_count"] = len(errors)
    if errors:
        analysis["issues"].append(f"{len(errors)} errors encountered")

    # Generate suggestions
    if not analysis["issues"]:
        analysis["suggestions"].append("Book finding completed successfully")
    else:
        if "Low search success rate" in str(analysis["issues"]):
            analysis["suggestions"].append("Some recommended books may not be available - this is normal")
        if "Low processing success rate" in str(analysis["issues"]):
            analysis["suggestions"].append("PDF processing can fail for various reasons - summaries still generated from recommendations")

    return analysis


def print_quality_analysis(analysis: dict) -> None:
    """Print quality analysis summary."""
    print("\n" + "=" * 80)
    print("QUALITY ANALYSIS")
    print("=" * 80)

    # Metrics
    print("\n--- Metrics ---")
    metrics = analysis.get("metrics", {})
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Issues
    issues = analysis.get("issues", [])
    if issues:
        print(f"\n--- Issues Found ({len(issues)}) ---")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n--- No Issues Found ---")

    # Suggestions
    suggestions = analysis.get("suggestions", [])
    if suggestions:
        print(f"\n--- Suggestions ---")
        for suggestion in suggestions:
            print(f"  - {suggestion}")

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
    parser = argparse.ArgumentParser(
        description="Run book finding workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "organizational resilience"              # Quick book finding
  %(prog)s "creative leadership" standard           # Standard quality
  %(prog)s "digital transformation" comprehensive   # Comprehensive search
  %(prog)s "liderazgo creativo" quick --language es # Spanish output
        """
    )

    parser.add_argument(
        "theme",
        nargs="?",
        default="The intersection of technology and human creativity",
        help="Theme to explore for book recommendations"
    )
    parser.add_argument(
        "quality",
        nargs="?",
        default=DEFAULT_QUALITY,
        choices=VALID_QUALITIES,
        help=f"Quality level (default: {DEFAULT_QUALITY})"
    )
    parser.add_argument(
        "--brief", "-b",
        type=str,
        default=None,
        help="Additional context to guide recommendations"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Language code for output (default: en). Supported: en, es, zh, ja, de, fr, pt, ko, ru, ar, it, nl, pl, tr, vi, th, id, hi, bn, sv, no, da, fi, cs, el, he, uk, ro, hu"
    )

    return parser.parse_args()


async def main():
    """Run book finding test."""
    args = parse_args()

    theme = args.theme
    quality = args.quality
    brief = args.brief
    language = args.language

    print(f"\n{'=' * 80}")
    print("BOOK FINDING TEST")
    print(f"{'=' * 80}")
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
        analysis = analyze_quality(result)
        print_quality_analysis(analysis)

        # Save results
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save full result
        result_file = OUTPUT_DIR / f"book_finding_result_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Full result saved to: {result_file}")

        # Save analysis
        analysis_file = OUTPUT_DIR / f"book_finding_analysis_{timestamp}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to: {analysis_file}")

        # Save final output as markdown
        if result.get("final_markdown"):
            report_file = OUTPUT_DIR / f"book_finding_{timestamp}.md"
            with open(report_file, "w") as f:
                f.write(f"# Book Recommendations: {theme}\n\n")
                f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write(f"*Quality: {quality} | Language: {language}*\n\n")
                f.write("---\n\n")
                f.write(result["final_markdown"])
            logger.info(f"Report saved to: {report_file}")

        return result, analysis

    except Exception as e:
        logger.error(f"Book finding failed: {e}", exc_info=True)
        return {"error": str(e)}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
