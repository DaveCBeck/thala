#!/usr/bin/env python3
"""
Test script for the Editing Workflow.

Runs structural editing on a document to improve coherence, organization,
and flow.

Usage:
    python testing/test_editing_workflow.py [input_file] [--quality QUALITY]
    python testing/test_editing_workflow.py  # Uses default test file
    python testing/test_editing_workflow.py my_document.md --quality standard

Environment:
    Set THALA_MODE=dev in .env to enable LangSmith tracing
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Enable dev mode for LangSmith tracing before any imports
os.environ["THALA_MODE"] = "dev"

import argparse
import logging
import langsmith as ls

from testing.utils import (
    configure_logging,
    get_output_dir,
    print_section_header,
    print_key_value,
    print_errors,
)
from workflows.enhance.editing import editing

configure_logging("editing_test")
logger = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

# Default test file (same as loop 3 test)
DEFAULT_INPUT = "testing/test_data/supervised_lit_review_loop2_20260114_133638.md"

VALID_QUALITIES = ["test", "quick", "standard", "comprehensive", "high_quality"]


def extract_topic_from_document(document: str) -> str:
    """Extract topic from document header or use default."""
    lines = document.split("\n")
    for line in lines:
        if line.startswith("# "):
            # Extract from title like "# Literature Review: Topic Name"
            title = line[2:].strip()
            if ":" in title:
                return title.split(":", 1)[1].strip()
            return title
    return "Unknown topic"


async def run_editing_test(
    input_file: str,
    quality: str = "test",
    topic: str | None = None,
) -> dict:
    """Run editing workflow test.

    Args:
        input_file: Path to markdown document file
        quality: Quality level for settings
        topic: Optional topic override

    Returns:
        Dict with test outputs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load input
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    document = input_path.read_text()
    detected_topic = extract_topic_from_document(document)
    final_topic = topic or detected_topic

    logger.info("=== Editing Workflow Test ===")
    logger.info(f"Input: {input_file}")
    logger.info(f"Topic: {final_topic}")
    logger.info(f"Quality: {quality}")
    logger.info(f"Input length: {len(document):,} chars")

    started_at = datetime.now()

    with ls.tracing_context(enabled=True):
        result = await editing(
            document=document,
            topic=final_topic,
            quality=quality,
        )

    completed_at = datetime.now()
    duration = (completed_at - started_at).total_seconds()

    logger.info("=== Editing Workflow Complete ===")
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"Status: {result['status']}")
    logger.info(f"Changes: {result.get('changes_summary', 'N/A')}")

    # Build output structure
    output = {
        "test_metadata": {
            "run_id": result["langsmith_run_id"],
            "timestamp": timestamp,
            "input_file": input_file,
            "topic": final_topic,
            "quality": quality,
            "input_length": len(document),
            "output_length": len(result.get("final_report", "")),
            "duration_seconds": duration,
            "status": result["status"],
            "changes_summary": result.get("changes_summary", ""),
            "structure_iterations": result.get("structure_iterations", 0),
        },
        "input_document": document,
        "output_document": result.get("final_report", ""),
        "errors": result.get("errors", []),
        "final_verification": result.get("final_verification", {}),
    }

    return output


def save_test_outputs(output: dict, timestamp: str) -> dict[str, str]:
    """Save test outputs in formats suitable for assessment.

    Returns:
        Dict mapping output type to file path
    """
    saved_files = {}

    # 1. Full JSON output (for programmatic analysis)
    json_path = OUTPUT_DIR / f"editing_test_full_{timestamp}.json"
    json_path.write_text(json.dumps(output, indent=2, default=str))
    saved_files["full_json"] = str(json_path)
    logger.info(f"Saved full output: {json_path}")

    # 2. Output document markdown
    output_md_path = OUTPUT_DIR / f"editing_test_output_{timestamp}.md"
    output_md_path.write_text(output["output_document"])
    saved_files["output_document"] = str(output_md_path)
    logger.info(f"Saved output document: {output_md_path}")

    # 3. Analysis summary
    analysis_path = OUTPUT_DIR / f"editing_test_analysis_{timestamp}.md"
    analysis_content = generate_analysis_summary(output)
    analysis_path.write_text(analysis_content)
    saved_files["analysis_summary"] = str(analysis_path)
    logger.info(f"Saved analysis summary: {analysis_path}")

    return saved_files


def generate_analysis_summary(output: dict) -> str:
    """Generate human-readable analysis summary."""
    meta = output["test_metadata"]
    lines = [
        "# Editing Workflow Test Analysis",
        "",
        f"**Run ID:** {meta['run_id']}",
        f"**Timestamp:** {meta['timestamp']}",
        f"**Topic:** {meta['topic']}",
        f"**Quality:** {meta['quality']}",
        f"**Duration:** {meta['duration_seconds']:.1f}s",
        f"**Status:** {meta['status']}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"**Input Length:** {meta['input_length']:,} chars",
        f"**Output Length:** {meta['output_length']:,} chars",
        f"**Structure Iterations:** {meta['structure_iterations']}",
        "",
        "### Changes Applied",
        "",
        meta.get("changes_summary", "No changes summary available."),
        "",
    ]

    # Final verification
    verification = output.get("final_verification", {})
    if verification:
        lines.extend([
            "## Final Verification Scores",
            "",
            f"- **Coherence:** {verification.get('coherence_score', 'N/A')}",
            f"- **Completeness:** {verification.get('completeness_score', 'N/A')}",
            f"- **Flow:** {verification.get('flow_score', 'N/A')}",
            f"- **Overall:** {verification.get('overall_score', 'N/A')}",
            "",
            f"**Has Introduction:** {verification.get('has_introduction', 'N/A')}",
            f"**Has Conclusion:** {verification.get('has_conclusion', 'N/A')}",
            f"**Well Organized:** {verification.get('sections_well_organized', 'N/A')}",
            "",
        ])

        remaining_issues = verification.get("remaining_issues", [])
        if remaining_issues:
            lines.append("### Remaining Issues")
            for issue in remaining_issues:
                lines.append(f"- {issue}")
            lines.append("")

        assessment = verification.get("overall_assessment", "")
        if assessment:
            lines.extend([
                "### Overall Assessment",
                "",
                assessment,
                "",
            ])

    # Errors
    errors = output.get("errors", [])
    if errors:
        lines.extend([
            "## Errors Encountered",
            "",
        ])
        for i, error in enumerate(errors, 1):
            lines.append(f"{i}. **{error.get('node', 'unknown')}:** {error.get('error', 'Unknown error')}")
        lines.append("")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="Test the Editing Workflow for structural improvements"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Path to markdown document file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--quality",
        choices=VALID_QUALITIES,
        default="test",
        help="Quality level for settings (default: test)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Override topic detection",
    )

    args = parser.parse_args()

    try:
        output = await run_editing_test(
            input_file=args.input_file,
            quality=args.quality,
            topic=args.topic,
        )

        timestamp = output["test_metadata"]["timestamp"]
        saved_files = save_test_outputs(output, timestamp)

        print("\n" + "=" * 60)
        print("EDITING WORKFLOW TEST COMPLETE")
        print("=" * 60)

        meta = output["test_metadata"]
        print_section_header("Test Results")
        print_key_value("Topic", meta["topic"])
        print_key_value("Status", meta["status"])
        print_key_value("Duration", f"{meta['duration_seconds']:.1f}s")
        print_key_value("Structure Iterations", meta["structure_iterations"])
        print_key_value("Input Length", f"{meta['input_length']:,} chars")
        print_key_value("Output Length", f"{meta['output_length']:,} chars")

        print_section_header("Changes Summary")
        print(meta.get("changes_summary", "No summary available."))

        # Print verification scores
        verification = output.get("final_verification", {})
        if verification:
            print_section_header("Verification Scores")
            print_key_value("Coherence", f"{verification.get('coherence_score', 0):.2f}")
            print_key_value("Completeness", f"{verification.get('completeness_score', 0):.2f}")
            print_key_value("Flow", f"{verification.get('flow_score', 0):.2f}")
            print_key_value("Overall", f"{verification.get('overall_score', 0):.2f}")

        print_section_header("Saved Files")
        for name, path in saved_files.items():
            print(f"  {name}: {path}")

        # Print errors if any
        if output.get("errors"):
            print_errors(output["errors"])

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
