#!/usr/bin/env python3
"""
Test script for the wrapped research workflow.

Runs the comprehensive research workflow that orchestrates web research,
academic literature review, and book finding with checkpointing support.

Usage:
    python test_wrapped_research.py "your research topic" [quality] [options]
    python test_wrapped_research.py "AI agents in creative work" quick
    python test_wrapped_research.py "AI agents in creative work" standard
    python test_wrapped_research.py  # Uses default topic and quick quality

Valid quality levels: quick, standard, comprehensive (default: quick)

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
LOG_FILE = LOG_DIR / f"wrapped_research_{_log_timestamp}.log"

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
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / "wrapped"

VALID_QUALITIES = ["quick", "standard", "comprehensive"]
DEFAULT_QUALITY = "quick"


# =============================================================================
# Checkpoint Utilities
# =============================================================================


def save_checkpoint(state: dict, name: str) -> Path:
    """Save workflow state to a checkpoint file for later resumption."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{name}.json"
    with open(checkpoint_file, "w") as f:
        json.dump(state, f, indent=2, default=str)
    logger.info(f"Checkpoint saved: {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(name: str) -> dict | None:
    """Load workflow state from a checkpoint file."""
    checkpoint_file = CHECKPOINT_DIR / f"{name}.json"
    if not checkpoint_file.exists():
        logger.error(f"Checkpoint not found: {checkpoint_file}")
        return None
    with open(checkpoint_file, "r") as f:
        state = json.load(f)
    logger.info(f"Checkpoint loaded: {checkpoint_file}")
    return state


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
        preview = output[:800]
        if len(output) > 800:
            preview += "\n  ... [truncated] ..."
        # Indent preview
        preview = "\n  ".join(preview.split("\n"))
        print(f"  Preview:\n  {preview}")
    else:
        error = result.get("error", "Unknown error")
        print(f"  Error: {error}")

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
            print(f"  Duration: {minutes}m {seconds}s")
        except Exception:
            pass


def print_result_summary(result: dict, query: str) -> None:
    """Print a detailed summary of the wrapped research result."""
    print("\n" + "=" * 80)
    print("WRAPPED RESEARCH RESULT")
    print("=" * 80)

    # Query and metadata
    print(f"\nQuery: {query}")

    # Overall timing
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
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            if hours > 0:
                print(f"Total Duration: {hours}h {minutes}m {seconds}s")
            else:
                print(f"Total Duration: {minutes}m {seconds}s ({duration:.1f}s total)")
        except Exception as e:
            print(f"Duration: (error calculating: {e})")

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
        # Show first 2000 chars
        preview = combined[:2000]
        if len(combined) > 2000:
            preview += "\n\n... [truncated] ..."
        print(preview)

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
    errors = result.get("errors", [])
    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for err in errors:
            phase = err.get("phase", "unknown")
            error = err.get("error", "unknown")
            print(f"  [{phase}]: {error}")

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

    return result


async def run_with_checkpoints(
    query: str,
    quality: str = "quick",
    research_questions: list[str] | None = None,
    date_range: tuple[int, int] | None = None,
    checkpoint_prefix: str = "latest",
) -> dict:
    """Run wrapped workflow with manual checkpoint saves for testing.

    The wrapped workflow has built-in checkpointing, but this adds explicit
    saves after each major phase for test iteration.

    Saves checkpoints:
    - {prefix}_after_parallel: After web + academic complete
    - {prefix}_after_books: After book finding complete
    - {prefix}_final: Complete result
    """
    from workflows.wrapped.state import (
        WrappedResearchState,
        WrappedResearchInput,
        CheckpointPhase,
        QUALITY_MAPPING,
    )
    from workflows.wrapped.nodes import (
        run_parallel_research,
        generate_book_query,
        run_book_finding,
        generate_final_summary,
        save_to_top_of_mind,
    )
    import uuid

    logger.info(f"Starting checkpointed wrapped research on: {query}")
    logger.info(f"Checkpoint prefix: {checkpoint_prefix}")

    run_id = str(uuid.uuid4())

    # Build initial state
    state = WrappedResearchState(
        input=WrappedResearchInput(
            query=query,
            quality=quality,
            research_questions=research_questions,
            date_range=date_range,
        ),
        web_result=None,
        academic_result=None,
        book_result=None,
        book_theme=None,
        book_brief=None,
        combined_summary=None,
        top_of_mind_ids={},
        checkpoint_phase=CheckpointPhase(
            parallel_research=False,
            book_query_generated=False,
            book_finding=False,
            saved_to_top_of_mind=False,
        ),
        checkpoint_path=None,
        started_at=datetime.utcnow(),
        completed_at=None,
        current_phase="starting",
        langsmith_run_id=run_id,
        errors=[],
    )

    # Phase 1: Parallel research (web + academic)
    logger.info("Running parallel research phase (web + academic)...")
    updates = await run_parallel_research(state)
    state = {**state, **updates}
    save_checkpoint(state, f"{checkpoint_prefix}_after_parallel")

    # Phase 2: Generate book query
    logger.info("Generating book query from research...")
    updates = await generate_book_query(state)
    state = {**state, **updates}

    # Phase 3: Book finding
    logger.info("Running book finding phase...")
    updates = await run_book_finding(state)
    state = {**state, **updates}
    save_checkpoint(state, f"{checkpoint_prefix}_after_books")

    # Phase 4: Generate final summary
    logger.info("Generating final summary...")
    updates = await generate_final_summary(state)
    state = {**state, **updates}

    # Phase 5: Save to top of mind
    logger.info("Saving to top of mind...")
    updates = await save_to_top_of_mind(state)
    state = {**state, **updates}

    state["completed_at"] = datetime.utcnow()
    save_checkpoint(state, f"{checkpoint_prefix}_final")

    return state


async def run_from_parallel_checkpoint(checkpoint_prefix: str) -> dict:
    """Resume workflow from after-parallel checkpoint.

    Runs: book_query -> book_finding -> summary -> save
    Skips: parallel research (web + academic)
    """
    from workflows.wrapped.nodes import (
        generate_book_query,
        run_book_finding,
        generate_final_summary,
        save_to_top_of_mind,
    )

    checkpoint_name = f"{checkpoint_prefix}_after_parallel"
    state = load_checkpoint(checkpoint_name)
    if not state:
        raise ValueError(f"Checkpoint not found: {checkpoint_name}")

    logger.info(f"Resuming from parallel checkpoint: {checkpoint_name}")

    # Phase 2: Generate book query
    logger.info("Generating book query from research...")
    updates = await generate_book_query(state)
    state = {**state, **updates}

    # Phase 3: Book finding
    logger.info("Running book finding phase...")
    updates = await run_book_finding(state)
    state = {**state, **updates}
    save_checkpoint(state, f"{checkpoint_prefix}_after_books")

    # Phase 4: Generate final summary
    logger.info("Generating final summary...")
    updates = await generate_final_summary(state)
    state = {**state, **updates}

    # Phase 5: Save to top of mind
    logger.info("Saving to top of mind...")
    updates = await save_to_top_of_mind(state)
    state = {**state, **updates}

    state["completed_at"] = datetime.utcnow()
    save_checkpoint(state, f"{checkpoint_prefix}_final")

    return state


async def run_from_books_checkpoint(checkpoint_prefix: str) -> dict:
    """Resume workflow from after-books checkpoint.

    Runs: summary -> save
    Skips: parallel research, book finding
    """
    from workflows.wrapped.nodes import (
        generate_final_summary,
        save_to_top_of_mind,
    )

    checkpoint_name = f"{checkpoint_prefix}_after_books"
    state = load_checkpoint(checkpoint_name)
    if not state:
        raise ValueError(f"Checkpoint not found: {checkpoint_name}")

    logger.info(f"Resuming from books checkpoint: {checkpoint_name}")

    # Phase 4: Generate final summary
    logger.info("Generating final summary...")
    updates = await generate_final_summary(state)
    state = {**state, **updates}

    # Phase 5: Save to top of mind
    logger.info("Saving to top of mind...")
    updates = await save_to_top_of_mind(state)
    state = {**state, **updates}

    state["completed_at"] = datetime.utcnow()
    save_checkpoint(state, f"{checkpoint_prefix}_final")

    return state


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run wrapped research workflow (web + academic + books)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "AI agents in creative work"              # Quick run with checkpoints
  %(prog)s "AI agents in creative work" standard     # Standard quality
  %(prog)s "Impact of LLMs" comprehensive            # Comprehensive (takes hours)

Checkpoint examples:
  %(prog)s "topic" quick --checkpoint-prefix mytest       # Full run, saves checkpoints
  %(prog)s --resume-from parallel --checkpoint-prefix mytest  # Resume from after parallel
  %(prog)s --resume-from books --checkpoint-prefix mytest     # Resume from after books
  %(prog)s "topic" quick --no-checkpoint                  # Original behavior (no checkpoints)
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        default="The impact of AI agents on knowledge work and creative processes",
        help="Research query/topic"
    )
    parser.add_argument(
        "quality",
        nargs="?",
        default=DEFAULT_QUALITY,
        choices=VALID_QUALITIES,
        help=f"Quality level (default: {DEFAULT_QUALITY})"
    )
    parser.add_argument(
        "--questions", "-q",
        type=str,
        nargs="+",
        default=None,
        help="Research questions for academic review (if not provided, will be auto-generated)"
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=None,
        help="Start year for academic date filter"
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=None,
        help="End year for academic date filter"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        choices=["parallel", "books"],
        default=None,
        help="Resume from a checkpoint (skips earlier phases)"
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="latest",
        help="Prefix for checkpoint files (default: 'latest')"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable manual checkpointing (use built-in workflow checkpoints only)"
    )

    return parser.parse_args()


async def main():
    """Run wrapped research workflow test."""
    args = parse_args()

    query = args.query
    quality = args.quality
    checkpoint_prefix = args.checkpoint_prefix

    # Research questions (optional, for academic workflow)
    research_questions = args.questions

    # Date range
    date_range = None
    if args.from_year or args.to_year:
        from_year = args.from_year or 2000
        to_year = args.to_year or 2025
        date_range = (from_year, to_year)

    # Determine run mode
    if args.resume_from:
        mode = f"resume from {args.resume_from}"
    elif args.no_checkpoint:
        mode = "no manual checkpoints"
    else:
        mode = "with manual checkpoints"

    print(f"\n{'=' * 80}")
    print("WRAPPED RESEARCH WORKFLOW TEST")
    print(f"{'=' * 80}")
    print(f"\nQuery: {query}")
    print(f"Quality: {quality}")
    print(f"Mode: {mode}")
    if research_questions:
        print(f"Research Questions:")
        for q in research_questions:
            print(f"  - {q}")
    if date_range:
        print(f"Date Range: {date_range[0]}-{date_range[1]}")
    if not args.no_checkpoint:
        print(f"Checkpoint prefix: {checkpoint_prefix}")
    print(f"LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'thala-dev')}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        # Choose run function based on mode
        if args.resume_from == "parallel":
            result = await run_from_parallel_checkpoint(checkpoint_prefix)
            # Extract query from checkpoint for display
            query = result.get("input", {}).get("query", query)
        elif args.resume_from == "books":
            result = await run_from_books_checkpoint(checkpoint_prefix)
            query = result.get("input", {}).get("query", query)
        elif args.no_checkpoint:
            result = await run_wrapped_research(
                query=query,
                quality=quality,
                research_questions=research_questions,
                date_range=date_range,
            )
        else:
            result = await run_with_checkpoints(
                query=query,
                quality=quality,
                research_questions=research_questions,
                date_range=date_range,
                checkpoint_prefix=checkpoint_prefix,
            )

        # Print detailed result summary
        print_result_summary(result, query)

        # Analyze quality
        analysis = analyze_quality(result)
        print_quality_analysis(analysis)

        # Save results
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save full result
        result_file = OUTPUT_DIR / f"wrapped_result_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Full result saved to: {result_file}")

        # Save analysis
        analysis_file = OUTPUT_DIR / f"wrapped_analysis_{timestamp}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to: {analysis_file}")

        # Save individual workflow outputs as markdown
        safe_query = query[:50].replace(" ", "_").replace("/", "-")

        # Web research
        web_result = result.get("web_result", {})
        if web_result and web_result.get("final_output"):
            web_file = OUTPUT_DIR / f"wrapped_web_{timestamp}.md"
            with open(web_file, "w") as f:
                f.write(f"# Web Research: {query}\n\n")
                f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write(f"*Quality: {quality}*\n\n")
                f.write("---\n\n")
                f.write(web_result["final_output"])
            logger.info(f"Web research saved to: {web_file}")

        # Academic review
        academic_result = result.get("academic_result", {})
        if academic_result and academic_result.get("final_output"):
            academic_file = OUTPUT_DIR / f"wrapped_academic_{timestamp}.md"
            with open(academic_file, "w") as f:
                f.write(f"# Academic Literature Review: {query}\n\n")
                f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write(f"*Quality: {quality}*\n\n")
                f.write("---\n\n")
                f.write(academic_result["final_output"])
            logger.info(f"Academic review saved to: {academic_file}")

        # Book recommendations
        book_result = result.get("book_result", {})
        if book_result and book_result.get("final_output"):
            book_file = OUTPUT_DIR / f"wrapped_books_{timestamp}.md"
            with open(book_file, "w") as f:
                f.write(f"# Book Recommendations: {query}\n\n")
                f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write(f"*Quality: {quality}*\n\n")
                f.write("---\n\n")
                f.write(book_result["final_output"])
            logger.info(f"Book recommendations saved to: {book_file}")

        # Combined summary
        combined = result.get("combined_summary")
        if combined:
            combined_file = OUTPUT_DIR / f"wrapped_combined_{timestamp}.md"
            with open(combined_file, "w") as f:
                f.write(f"# Combined Research Summary: {query}\n\n")
                f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write(f"*Quality: {quality}*\n\n")
                f.write("---\n\n")
                f.write(combined)
            logger.info(f"Combined summary saved to: {combined_file}")

        return result, analysis

    except Exception as e:
        logger.error(f"Wrapped research failed: {e}", exc_info=True)
        return {"error": str(e)}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
