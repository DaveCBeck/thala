#!/usr/bin/env python3
"""
Test script for running supervision workflow on an existing lit review result.

Takes a lit review result.json and runs the enhancement supervision loops
(Loop 1: theoretical depth, Loop 2: literature expansion).

Usage:
    python test_supervision_from_result.py path/to/lit_review_result.json [options]
    python test_supervision_from_result.py testing/test_data/lit_review_result_20260116_231835.json
    python test_supervision_from_result.py result.json --quality quick --loops one

Options:
    --quality    Quality tier: quick, standard, comprehensive, high_quality (default: quick)
    --loops      Which loops to run: none, one, two, all (default: all)
    --max-iter   Max iterations per loop (default: 3)

Environment:
    Set THALA_MODE=dev in .env to enable LangSmith tracing
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

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
)

configure_logging("supervision")
logger = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

VALID_QUALITIES = ["quick", "standard", "comprehensive", "high_quality"]
VALID_LOOPS = ["none", "one", "two", "all"]


def load_lit_review_result(path: str) -> dict:
    """Load a lit review result.json file."""
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    with open(result_path) as f:
        return json.load(f)


def extract_supervision_inputs(result: dict) -> tuple[str, str, list[str]]:
    """Extract the three required inputs from a lit review result.

    Returns:
        Tuple of (report, topic, research_questions)
    """
    # Get report - try final_review first, then final_report
    report = result.get("final_review") or result.get("final_report")
    if not report:
        raise ValueError("Result has no 'final_review' or 'final_report' field")

    # Get input fields
    input_data = result.get("input", {})
    topic = input_data.get("topic")
    if not topic:
        raise ValueError("Result has no 'input.topic' field")

    research_questions = input_data.get("research_questions", [])
    if not research_questions:
        raise ValueError("Result has no 'input.research_questions' field")

    return report, topic, research_questions


def print_result_summary(result: dict, topic: str) -> None:
    """Print a summary of the supervision result."""
    print_section_header("SUPERVISION WORKFLOW RESULT")

    print(f"\nTopic: {topic}")
    print(f"Completion: {result.get('completion_reason', 'unknown')}")
    print(f"Loops run: {result.get('loops_run', [])}")

    # Report lengths
    print("\n--- Report Lengths ---")
    if result.get("review_loop1"):
        print(f"After Loop 1: {len(result['review_loop1'])} chars")
    if result.get("review_loop2"):
        print(f"After Loop 2: {len(result['review_loop2'])} chars")

    final_report = result.get("final_report", "")
    if final_report:
        word_count = len(final_report.split())
        print(f"Final report: {len(final_report)} chars ({word_count} words)")

    # Papers found during supervision
    paper_corpus = result.get("paper_corpus", {})
    paper_summaries = result.get("paper_summaries", {})
    zotero_keys = result.get("zotero_keys", {})

    print("\n--- Papers (from supervision) ---")
    print(f"Papers in corpus: {len(paper_corpus)}")
    print(f"Papers summarized: {len(paper_summaries)}")
    print(f"Zotero items: {len(zotero_keys)}")

    # Preview final report
    if final_report:
        print("\n--- Final Report Preview ---")
        print(safe_preview(final_report, 1500))

    # Errors
    print_errors(result.get("errors", []))

    print("\n" + "=" * 80)


async def run_supervision(
    report: str,
    topic: str,
    research_questions: list[str],
    quality: str = "quick",
    loops: str = "all",
    max_iterations: int = 3,
) -> dict:
    """Run the supervision workflow."""
    from workflows.enhance.supervision.api import enhance_report

    logger.info(f"Starting supervision on: {topic}")
    logger.info(f"Quality: {quality}, Loops: {loops}, Max iterations: {max_iterations}")
    logger.info(f"Input report length: {len(report)} chars")

    result = await enhance_report(
        report=report,
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        loops=loops,
        max_iterations_per_loop=max_iterations,
    )

    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run supervision workflow on a lit review result",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s testing/test_data/lit_review_result.json
  %(prog)s result.json --quality standard
  %(prog)s result.json --loops one --max-iter 2
  %(prog)s result.json --quality comprehensive --loops all
        """,
    )

    parser.add_argument(
        "result_file",
        type=str,
        help="Path to lit review result.json file",
    )

    parser.add_argument(
        "--quality",
        type=str,
        choices=VALID_QUALITIES,
        default="quick",
        help="Quality tier (default: quick)",
    )

    parser.add_argument(
        "--loops",
        type=str,
        choices=VALID_LOOPS,
        default="all",
        help="Which loops to run (default: all)",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Max iterations per loop (default: 3)",
    )

    return parser.parse_args()


async def main():
    """Run supervision workflow test."""
    args = parse_args()

    # Load the lit review result
    print_section_header("LOADING LIT REVIEW RESULT")
    print(f"\nResult file: {args.result_file}")

    try:
        result_data = load_lit_review_result(args.result_file)
        report, topic, research_questions = extract_supervision_inputs(result_data)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load result: {e}")
        sys.exit(1)

    print(f"Topic: {topic}")
    print(f"Research questions: {len(research_questions)}")
    print(f"Report length: {len(report)} chars ({len(report.split())} words)")

    # Run supervision
    print_section_header("SUPERVISION WORKFLOW TEST")
    print(f"\nQuality: {args.quality}")
    print(f"Loops: {args.loops}")
    print(f"Max iterations: {args.max_iter}")
    print("Research Questions:")
    for q in research_questions:
        print(f"  - {q[:80]}{'...' if len(q) > 80 else ''}")
    print(f"LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'thala-dev')}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    start_time = datetime.now()

    try:
        result = await run_supervision(
            report=report,
            topic=topic,
            research_questions=research_questions,
            quality=args.quality,
            loops=args.loops,
            max_iterations=args.max_iter,
        )

        end_time = datetime.now()
        result["started_at"] = start_time.isoformat()
        result["completed_at"] = end_time.isoformat()

        # Print result summary
        print_result_summary(result, topic)
        print_timing(start_time.isoformat(), end_time.isoformat())

        # Save results
        result_file = save_json_result(result, "supervision_result")
        logger.info(f"Full result saved to: {result_file}")

        # Save final report as markdown
        if result.get("final_report"):
            report_file = save_markdown_report(
                result["final_report"],
                "supervision",
                title=f"Supervised Enhancement: {topic}",
                metadata={
                    "quality": args.quality,
                    "loops": args.loops,
                    "loops_run": result.get("loops_run", []),
                },
            )
            logger.info(f"Report saved to: {report_file}")

        # Save intermediate reports if available
        if result.get("review_loop1"):
            save_markdown_report(
                result["review_loop1"],
                "supervision_loop1",
                title=f"After Loop 1: {topic}",
            )

        if result.get("review_loop2"):
            save_markdown_report(
                result["review_loop2"],
                "supervision_loop2",
                title=f"After Loop 2: {topic}",
            )

        return result

    except Exception as e:
        logger.error(f"Supervision test failed: {e}", exc_info=True)
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
