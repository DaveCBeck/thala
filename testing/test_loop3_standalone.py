#!/usr/bin/env python3
"""
Test script for Loop 3 (Structure & Cohesion) standalone execution.

Runs Loop 3 on a simulated Loop 2 output to test structural issue
identification and resolution.

Usage:
    python testing/test_loop3_standalone.py [input_file] [--quality QUALITY]
    python testing/test_loop3_standalone.py  # Uses default test file
    python testing/test_loop3_standalone.py testing/test_data/my_loop2_output.md --quality standard

Environment:
    Set THALA_MODE=dev in .env to enable LangSmith tracing
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

# Enable dev mode for LangSmith tracing before any imports
os.environ["THALA_MODE"] = "dev"

import argparse
import logging

import langsmith as ls
from langchain_core.tracers.langchain import wait_for_all_tracers
from workflows.wrappers.supervised_lit_review.supervision.loops.loop3.graph import (
    run_loop3_standalone,
)

from testing.utils import configure_logging, get_output_dir
from workflows.research.academic_lit_review.quality_presets import QUALITY_PRESETS

configure_logging("loop3_test")
logger = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

# Default test file
DEFAULT_INPUT = "testing/test_data/supervised_lit_review_loop2_20260114_133638.md"

VALID_QUALITIES = ["test", "quick", "standard", "comprehensive", "high_quality"]


def extract_topic_from_review(review_text: str) -> str:
    """Extract topic from review header or use default."""
    lines = review_text.split("\n")
    for line in lines:
        if line.startswith("# Literature Review"):
            # Extract from title like "# Literature Review: Topic Name"
            if ":" in line:
                return line.split(":", 1)[1].strip()
    return "Unknown topic"


async def run_loop3_test(
    input_file: str,
    quality: str = "test",
    max_iterations: int | None = None,
) -> dict:
    """Run Loop 3 standalone test.

    Args:
        input_file: Path to loop 2 output markdown file
        quality: Quality level for settings
        max_iterations: Override max iterations if specified

    Returns:
        Dict with all test outputs for assessment
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4()

    # Load input
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    review_text = input_path.read_text()
    topic = extract_topic_from_review(review_text)

    logger.info(f"=== Loop 3 Standalone Test ===")
    logger.info(f"Input: {input_file}")
    logger.info(f"Topic: {topic}")
    logger.info(f"Quality: {quality}")
    logger.info(f"Input length: {len(review_text)} chars")

    # Get quality settings
    quality_settings = dict(QUALITY_PRESETS.get(quality, QUALITY_PRESETS["test"]))
    if max_iterations is not None:
        quality_settings["max_stages"] = max_iterations - 1  # Loop 3 adds 1

    logger.info(f"Max iterations: {quality_settings.get('max_stages', 3) + 1}")

    # Run with tracing
    config = {
        "run_id": run_id,
        "run_name": f"loop3_test:{topic[:30]}",
    }

    started_at = datetime.now()

    with ls.tracing_context(enabled=True):
        result = await run_loop3_standalone(
            review=review_text,
            topic=topic,
            quality_settings=quality_settings,
            config=config,
        )

    completed_at = datetime.now()
    duration = (completed_at - started_at).total_seconds()

    logger.info(f"=== Loop 3 Complete ===")
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"Changes: {result.changes_summary}")
    logger.info(f"Iterations used: {result.iterations_used}")

    # Build output structure
    output = {
        "test_metadata": {
            "run_id": str(run_id),
            "timestamp": timestamp,
            "input_file": input_file,
            "topic": topic,
            "quality": quality,
            "input_length": len(review_text),
            "output_length": len(result.current_review),
            "duration_seconds": duration,
            "iterations_used": result.iterations_used,
            "changes_summary": result.changes_summary,
        },
        "input_review": review_text,
        "output_review": result.current_review,
    }

    # Load saved state for detailed analysis
    state_path = Path.home() / ".thala" / "workflow_states" / "supervision_loop3" / f"{run_id}.json"
    if state_path.exists():
        state_data = json.loads(state_path.read_text())
        output["saved_state"] = state_data

        # Extract key fields for assessment
        final_state = state_data.get("final_state", {})
        output["issue_analysis"] = state_data.get("input", {}).get("issue_analysis")
        output["rewrite_manifest"] = final_state.get("rewrite_manifest", {})
        output["architecture_verification"] = final_state.get("architecture_verification", {})
        output["changes_applied"] = final_state.get("changes_applied", [])

        logger.info(f"Loaded state from: {state_path}")
    else:
        logger.warning(f"State file not found: {state_path}")

    return output


def save_test_outputs(output: dict, timestamp: str) -> dict[str, str]:
    """Save test outputs in formats suitable for assessment.

    Returns:
        Dict mapping output type to file path
    """
    saved_files = {}

    # 1. Full JSON output (for programmatic analysis)
    json_path = OUTPUT_DIR / f"loop3_test_full_{timestamp}.json"
    json_path.write_text(json.dumps(output, indent=2, default=str))
    saved_files["full_json"] = str(json_path)
    logger.info(f"Saved full output: {json_path}")

    # 2. Output review markdown
    output_md_path = OUTPUT_DIR / f"loop3_test_output_{timestamp}.md"
    output_md_path.write_text(output["output_review"])
    saved_files["output_review"] = str(output_md_path)
    logger.info(f"Saved output review: {output_md_path}")

    # 3. Issue analysis summary (for easy reading)
    analysis_path = OUTPUT_DIR / f"loop3_test_analysis_{timestamp}.md"
    analysis_content = generate_analysis_summary(output)
    analysis_path.write_text(analysis_content)
    saved_files["analysis_summary"] = str(analysis_path)
    logger.info(f"Saved analysis summary: {analysis_path}")

    return saved_files


def generate_analysis_summary(output: dict) -> str:
    """Generate human-readable analysis summary for exploration agents."""
    lines = [
        "# Loop 3 Test Analysis Summary",
        "",
        f"**Run ID:** {output['test_metadata']['run_id']}",
        f"**Timestamp:** {output['test_metadata']['timestamp']}",
        f"**Topic:** {output['test_metadata']['topic']}",
        f"**Duration:** {output['test_metadata']['duration_seconds']:.1f}s",
        f"**Iterations:** {output['test_metadata']['iterations_used']}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"**Changes Summary:** {output['test_metadata']['changes_summary']}",
        f"**Input Length:** {output['test_metadata']['input_length']:,} chars",
        f"**Output Length:** {output['test_metadata']['output_length']:,} chars",
        "",
    ]

    # Rewrite manifest details
    manifest = output.get("rewrite_manifest", {})
    if manifest:
        lines.extend([
            "## Rewrite Manifest",
            "",
            f"**Issues Addressed:** {manifest.get('issues_addressed', [])}",
            f"**Issues Skipped:** {manifest.get('issues_skipped', [])}",
            "",
        ])

        skip_reasons = manifest.get("skip_reasons", {})
        if skip_reasons:
            lines.append("### Skip Reasons")
            for issue_id, reason in skip_reasons.items():
                lines.append(f"- Issue {issue_id}: {reason}")
            lines.append("")

        rewrites = manifest.get("rewrites", [])
        if rewrites:
            lines.append("### Rewrites Applied")
            for i, rewrite in enumerate(rewrites, 1):
                lines.extend([
                    f"",
                    f"#### Rewrite {i} (Issue {rewrite.get('issue_id', 'N/A')})",
                    f"- **Paragraphs:** {rewrite.get('original_paragraphs', [])}",
                    f"- **Confidence:** {rewrite.get('confidence', 'N/A')}",
                    f"- **Summary:** {rewrite.get('changes_summary', 'N/A')}",
                ])
            lines.append("")

    # Architecture verification
    verification = output.get("architecture_verification", {})
    if verification:
        lines.extend([
            "## Architecture Verification",
            "",
            f"**Coherence Score:** {verification.get('coherence_score', 'N/A')}",
            f"**Needs Another Iteration:** {verification.get('needs_another_iteration', 'N/A')}",
            "",
        ])

        resolved = verification.get("issues_resolved", [])
        if resolved:
            lines.append("### Issues Resolved")
            for issue in resolved:
                lines.append(f"- {issue}")
            lines.append("")

        remaining = verification.get("issues_remaining", [])
        if remaining:
            lines.append("### Issues Remaining")
            for issue in remaining:
                lines.append(f"- {issue}")
            lines.append("")

        regressions = verification.get("regressions_introduced", [])
        if regressions:
            lines.append("### Regressions Introduced")
            for issue in regressions:
                lines.append(f"- {issue}")
            lines.append("")

        reasoning = verification.get("reasoning", "")
        if reasoning:
            lines.extend([
                "### Reasoning",
                "",
                reasoning,
                "",
            ])

    # Changes applied
    changes = output.get("changes_applied", [])
    if changes:
        lines.extend([
            "## Changes Applied",
            "",
        ])
        for i, change in enumerate(changes, 1):
            lines.append(f"{i}. {change}")
        lines.append("")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="Test Loop 3 (Structure & Cohesion) standalone"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Path to loop 2 output markdown file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--quality",
        choices=VALID_QUALITIES,
        default="test",
        help="Quality level for settings (default: test)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Override max iterations",
    )

    args = parser.parse_args()

    try:
        output = await run_loop3_test(
            input_file=args.input_file,
            quality=args.quality,
            max_iterations=args.max_iterations,
        )

        timestamp = output["test_metadata"]["timestamp"]
        saved_files = save_test_outputs(output, timestamp)

        print("\n" + "=" * 60)
        print("LOOP 3 TEST COMPLETE")
        print("=" * 60)
        print(f"\nTopic: {output['test_metadata']['topic']}")
        print(f"Duration: {output['test_metadata']['duration_seconds']:.1f}s")
        print(f"Iterations: {output['test_metadata']['iterations_used']}")
        print(f"Changes: {output['test_metadata']['changes_summary']}")
        print("\nSaved files:")
        for name, path in saved_files.items():
            print(f"  {name}: {path}")

        # Print verification summary
        verification = output.get("architecture_verification", {})
        if verification:
            print(f"\nArchitecture Verification:")
            print(f"  Coherence Score: {verification.get('coherence_score', 'N/A')}")
            print(f"  Issues Resolved: {len(verification.get('issues_resolved', []))}")
            print(f"  Issues Remaining: {len(verification.get('issues_remaining', []))}")
            print(f"  Regressions: {len(verification.get('regressions_introduced', []))}")

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Wait for LangSmith to flush all trace data before exiting
        wait_for_all_tracers()
