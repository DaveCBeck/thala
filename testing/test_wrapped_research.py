#!/usr/bin/env python3
"""
Test script for the wrapped research workflow.

Runs the comprehensive research workflow that orchestrates web research,
academic literature review, and book finding.

Usage:
    python test_wrapped_research.py "your research topic" [quality] [options]
    python test_wrapped_research.py "AI agents in creative work" quick
    python test_wrapped_research.py "AI agents in creative work" standard
    python test_wrapped_research.py  # Uses default topic and quick quality

Valid quality levels: quick, standard, comprehensive (default: quick)

Environment:
    Set THALA_MODE=dev in .env to enable LangSmith tracing
"""

import asyncio
import os
from datetime import datetime

# Enable dev mode for LangSmith tracing before any imports
os.environ["THALA_MODE"] = "dev"

import logging

from testing.utils import (
    configure_logging,
    get_output_dir,
    save_json_result,
    save_markdown_report,
    print_section_header,
    safe_preview,
    print_timing,
    print_errors,
    print_quality_analysis,
    create_test_parser,
    add_quality_argument,
    add_date_range_arguments,
    add_research_questions_argument,
)
from workflows.shared.workflow_state_store import load_workflow_state

configure_logging("wrapped_research")
logger = logging.getLogger(__name__)

# Output directory for results
OUTPUT_DIR = get_output_dir()

VALID_QUALITIES = ["quick", "standard", "comprehensive"]
DEFAULT_QUALITY = "quick"


# =============================================================================
# Result Display
# =============================================================================


def print_workflow_result_summary(result: dict, workflow_type: str) -> None:
    """Print summary for a single workflow result."""
    if not result:
        print(f"  {workflow_type}: No result")
        return

    status = result.get("status", "unknown")
    output = result.get("final_output")

    print(f"\n  --- {workflow_type.upper()} ({status}) ---")

    if output:
        word_count = len(output.split())
        char_count = len(output)
        print(f"  Length: {char_count:,} chars ({word_count:,} words)")

        # Show preview
        preview = safe_preview(output, 800)
        # Indent preview
        preview = "\n  ".join(preview.split("\n"))
        print(f"  Preview:\n  {preview}")
    else:
        error = result.get("error", "Unknown error")
        print(f"  Error: {error}")

    # Timing
    print_timing(result.get("started_at"), result.get("completed_at"))


def print_result_summary(result: dict, query: str) -> None:
    """Print a detailed summary of the wrapped research result."""
    print_section_header("WRAPPED RESEARCH RESULT")

    # Query and metadata
    print(f"\nQuery: {query}")

    # Overall timing
    print_timing(result.get("started_at"), result.get("completed_at"))

    # Individual workflow results
    print("\n" + "-" * 80)
    print("WORKFLOW RESULTS")
    print("-" * 80)

    print_workflow_result_summary(result.get("web_result"), "Web Research")
    print_workflow_result_summary(result.get("academic_result"), "Academic Literature")
    print_workflow_result_summary(result.get("book_result"), "Book Recommendations")

    # Combined summary
    combined = result.get("combined_summary")
    if combined:
        print("\n" + "-" * 80)
        print("COMBINED SUMMARY")
        print("-" * 80)
        word_count = len(combined.split())
        print(f"Length: {len(combined):,} chars ({word_count:,} words)")
        print(safe_preview(combined, 2000))

    # Top of Mind IDs
    top_of_mind_ids = result.get("top_of_mind_ids", {})
    if top_of_mind_ids:
        print("\n" + "-" * 80)
        print("TOP OF MIND RECORDS")
        print("-" * 80)
        for workflow, record_id in top_of_mind_ids.items():
            print(f"  {workflow}: {record_id}")

    # LangSmith
    langsmith_run_id = result.get("langsmith_run_id")
    if langsmith_run_id:
        print(f"\nLangSmith Run ID: {langsmith_run_id}")

    # Errors
    print_errors(result.get("errors", []))

    print("\n" + "=" * 80)


def analyze_quality(result: dict) -> dict:
    """Analyze wrapped research quality and return metrics."""
    analysis = {
        "metrics": {},
        "issues": [],
        "suggestions": [],
    }

    # Check each workflow
    workflows = [
        ("web", result.get("web_result")),
        ("academic", result.get("academic_result")),
        ("books", result.get("book_result")),
    ]

    completed_count = 0
    for name, wf_result in workflows:
        if wf_result:
            status = wf_result.get("status", "unknown")
            analysis["metrics"][f"{name}_status"] = status
            if status == "completed":
                completed_count += 1
                output = wf_result.get("final_output", "")
                analysis["metrics"][f"{name}_word_count"] = len(output.split()) if output else 0
            else:
                analysis["issues"].append(f"{name} workflow failed: {wf_result.get('error', 'unknown')}")
        else:
            analysis["metrics"][f"{name}_status"] = "missing"
            analysis["issues"].append(f"{name} workflow result missing")

    analysis["metrics"]["workflows_completed"] = completed_count
    analysis["metrics"]["workflows_total"] = 3

    # Combined summary
    combined = result.get("combined_summary", "")
    if combined:
        analysis["metrics"]["combined_word_count"] = len(combined.split())
        analysis["metrics"]["combined_char_count"] = len(combined)
        analysis["metrics"]["has_combined_summary"] = True
    else:
        analysis["metrics"]["has_combined_summary"] = False
        analysis["issues"].append("No combined summary generated")

    # Top of mind
    top_of_mind_ids = result.get("top_of_mind_ids", {})
    analysis["metrics"]["records_saved"] = len(top_of_mind_ids)

    # Errors
    errors = result.get("errors", [])
    analysis["metrics"]["error_count"] = len(errors)
    if errors:
        analysis["issues"].append(f"{len(errors)} errors encountered")

    # Generate suggestions
    if completed_count == 3 and not analysis["issues"]:
        analysis["suggestions"].append("All workflows completed successfully")
    else:
        if completed_count < 3:
            analysis["suggestions"].append(f"Only {completed_count}/3 workflows completed - check service availability")
        if not combined:
            analysis["suggestions"].append("Check LLM availability for summary generation")

    return analysis


# =============================================================================
# Workflow Execution
# =============================================================================


async def run_wrapped_research(
    query: str,
    quality: str = "quick",
    research_questions: list[str] | None = None,
    date_range: tuple[int, int] | None = None,
) -> dict:
    """Run the wrapped research workflow."""
    from workflows.wrapped import wrapped_research

    logger.info(f"Starting wrapped research on: {query}")
    logger.info(f"Quality: {quality}")
    logger.info(f"LangSmith tracing: {os.environ.get('LANGSMITH_TRACING', 'false')}")

    result = await wrapped_research(
        query=query,
        quality=quality,
        research_questions=research_questions,
        date_range=date_range,
    )

    # Load full state from state store for detailed analysis
    run_id = result.get("langsmith_run_id")
    if run_id:
        full_state = load_workflow_state("wrapped_research", run_id)
        if full_state:
            result = {**full_state, **result}
            logger.info(f"Loaded full state from state store for run {run_id}")
        else:
            logger.warning(f"Could not load state for run {run_id} - detailed metrics unavailable")

    return result


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = create_test_parser(
        description="Run wrapped research workflow (web + academic + books)",
        default_topic="The impact of AI agents on knowledge work and creative processes",
        topic_help="Research query/topic",
        epilog_examples="""
Examples:
  %(prog)s "AI agents in creative work"              # Quick run
  %(prog)s "AI agents in creative work" standard     # Standard quality
  %(prog)s "Impact of LLMs" comprehensive            # Comprehensive (takes hours)
        """
    )

    add_quality_argument(parser, choices=VALID_QUALITIES, default=DEFAULT_QUALITY)
    add_research_questions_argument(parser)
    add_date_range_arguments(parser)

    return parser.parse_args()


async def main():
    """Run wrapped research workflow test."""
    args = parse_args()

    query = args.topic  # Note: topic arg is used as query
    quality = args.quality

    # Research questions (optional, for academic workflow)
    research_questions = args.questions

    # Date range
    date_range = None
    if args.from_year or args.to_year:
        from_year = args.from_year or 2000
        to_year = args.to_year or 2025
        date_range = (from_year, to_year)

    print_section_header("WRAPPED RESEARCH WORKFLOW TEST")
    print(f"\nQuery: {query}")
    print(f"Quality: {quality}")
    if research_questions:
        print(f"Research Questions:")
        for q in research_questions:
            print(f"  - {q}")
    if date_range:
        print(f"Date Range: {date_range[0]}-{date_range[1]}")
    print(f"LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'thala-dev')}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        result = await run_wrapped_research(
            query=query,
            quality=quality,
            research_questions=research_questions,
            date_range=date_range,
        )

        # Print detailed result summary
        print_result_summary(result, query)

        # Analyze quality
        analysis = analyze_quality(result)
        print_quality_analysis(analysis)

        # Save results
        result_file = save_json_result(result, "wrapped_result")
        logger.info(f"Full result saved to: {result_file}")

        analysis_file = save_json_result(analysis, "wrapped_analysis")
        logger.info(f"Analysis saved to: {analysis_file}")

        # Save individual workflow outputs as markdown
        # Web research
        web_result = result.get("web_result", {})
        if web_result and web_result.get("final_output"):
            web_file = save_markdown_report(
                web_result["final_output"],
                "wrapped_web",
                title=f"Web Research: {query}",
                metadata={"quality": quality},
            )
            logger.info(f"Web research saved to: {web_file}")

        # Academic review
        academic_result = result.get("academic_result", {})
        if academic_result and academic_result.get("final_output"):
            academic_file = save_markdown_report(
                academic_result["final_output"],
                "wrapped_academic",
                title=f"Academic Literature Review: {query}",
                metadata={"quality": quality},
            )
            logger.info(f"Academic review saved to: {academic_file}")

        # Book recommendations
        book_result = result.get("book_result", {})
        if book_result and book_result.get("final_output"):
            book_file = save_markdown_report(
                book_result["final_output"],
                "wrapped_books",
                title=f"Book Recommendations: {query}",
                metadata={"quality": quality},
            )
            logger.info(f"Book recommendations saved to: {book_file}")

        # Combined summary
        combined = result.get("combined_summary")
        if combined:
            combined_file = save_markdown_report(
                combined,
                "wrapped_combined",
                title=f"Combined Research Summary: {query}",
                metadata={"quality": quality},
            )
            logger.info(f"Combined summary saved to: {combined_file}")

        return result, analysis

    except Exception as e:
        logger.error(f"Wrapped research failed: {e}", exc_info=True)
        return {"error": str(e)}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
