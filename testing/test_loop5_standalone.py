#!/usr/bin/env python3
"""
Test script for Loop 5 (Fact & Reference Checking) standalone execution.

Runs Loop 5 on a simulated Loop 4 output to test fact checking, reference
verification, citation validation, and TODO filtering.

Usage:
    python testing/test_loop5_standalone.py [input_file] [--quality QUALITY]
    python testing/test_loop5_standalone.py  # Uses default test file
    python testing/test_loop5_standalone.py testing/test_data/my_loop4_output.md --quality standard

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
from workflows.wrappers.supervised_lit_review.supervision.loops.loop5.graph import (
    run_loop5_standalone,
)

from testing.utils import configure_logging, get_output_dir
from workflows.research.academic_lit_review.quality_presets import QUALITY_PRESETS

configure_logging("loop5_test")
logger = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

# Default test file
DEFAULT_INPUT = "testing/test_data/supervised_lit_review_loop4_20260114_133638.md"

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


async def run_loop5_test(
    input_file: str,
    quality: str = "test",
) -> dict:
    """Run Loop 5 standalone test.

    Args:
        input_file: Path to loop 4 output markdown file
        quality: Quality level for settings

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

    logger.info(f"=== Loop 5 Standalone Test ===")
    logger.info(f"Input: {input_file}")
    logger.info(f"Topic: {topic}")
    logger.info(f"Quality: {quality}")
    logger.info(f"Input length: {len(review_text)} chars")

    # Get quality settings
    quality_settings = dict(QUALITY_PRESETS.get(quality, QUALITY_PRESETS["test"]))

    logger.info(f"Max stages: {quality_settings.get('max_stages', 1)}")

    # Run with tracing
    config = {
        "run_id": run_id,
        "run_name": f"loop5_test:{topic[:30]}",
    }

    started_at = datetime.now()

    with ls.tracing_context(enabled=True):
        result = await run_loop5_standalone(
            review=review_text,
            topic=topic,
            quality_settings=quality_settings,
            config=config,
        )

    completed_at = datetime.now()
    duration = (completed_at - started_at).total_seconds()

    logger.info(f"=== Loop 5 Complete ===")
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"Changes: {result.changes_summary}")
    logger.info(f"Human review items: {len(result.human_review_items)}")

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
            "changes_summary": result.changes_summary,
            "human_review_items_count": len(result.human_review_items),
        },
        "input_review": review_text,
        "output_review": result.current_review,
        "human_review_items": result.human_review_items,
    }

    # Load saved state for detailed analysis
    state_path = Path.home() / ".thala" / "workflow_states" / "supervision_loop5" / f"{run_id}.json"
    if state_path.exists():
        state_data = json.loads(state_path.read_text())
        output["saved_state"] = state_data
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
    json_path = OUTPUT_DIR / f"loop5_test_full_{timestamp}.json"
    json_path.write_text(json.dumps(output, indent=2, default=str))
    saved_files["full_json"] = str(json_path)
    logger.info(f"Saved full output: {json_path}")

    # 2. Output review markdown
    output_md_path = OUTPUT_DIR / f"loop5_test_output_{timestamp}.md"
    output_md_path.write_text(output["output_review"])
    saved_files["output_review"] = str(output_md_path)
    logger.info(f"Saved output review: {output_md_path}")

    # 3. Analysis summary (for easy reading)
    analysis_path = OUTPUT_DIR / f"loop5_test_analysis_{timestamp}.md"
    analysis_content = generate_analysis_summary(output)
    analysis_path.write_text(analysis_content)
    saved_files["analysis_summary"] = str(analysis_path)
    logger.info(f"Saved analysis summary: {analysis_path}")

    # 4. Human review items
    if output.get("human_review_items"):
        review_items_path = OUTPUT_DIR / f"loop5_test_human_review_{timestamp}.md"
        review_content = generate_human_review_report(output)
        review_items_path.write_text(review_content)
        saved_files["human_review"] = str(review_items_path)
        logger.info(f"Saved human review items: {review_items_path}")

    return saved_files


def generate_analysis_summary(output: dict) -> str:
    """Generate human-readable analysis summary for exploration agents."""
    lines = [
        "# Loop 5 Test Analysis Summary",
        "",
        f"**Run ID:** {output['test_metadata']['run_id']}",
        f"**Timestamp:** {output['test_metadata']['timestamp']}",
        f"**Topic:** {output['test_metadata']['topic']}",
        f"**Duration:** {output['test_metadata']['duration_seconds']:.1f}s",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"**Changes Summary:** {output['test_metadata']['changes_summary']}",
        f"**Input Length:** {output['test_metadata']['input_length']:,} chars",
        f"**Output Length:** {output['test_metadata']['output_length']:,} chars",
        f"**Human Review Items:** {output['test_metadata']['human_review_items_count']}",
        "",
    ]

    # Saved state details
    state = output.get("saved_state", {})
    if state:
        final_state = state.get("final_state", {})
        lines.extend([
            "## State Analysis",
            "",
            f"**Valid Edits:** {final_state.get('valid_edits', 'N/A')}",
            f"**Invalid Edits:** {final_state.get('invalid_edits', 'N/A')}",
            f"**Ambiguous Claims:** {final_state.get('ambiguous_claims', 'N/A')}",
            f"**Discarded TODOs:** {final_state.get('discarded_todos', 'N/A')}",
            "",
        ])

    # Human review items summary
    human_items = output.get("human_review_items", [])
    if human_items:
        lines.extend([
            "## Human Review Items Preview",
            "",
        ])
        for i, item in enumerate(human_items[:10], 1):
            # Truncate long items
            item_text = str(item)[:200]
            if len(str(item)) > 200:
                item_text += "..."
            lines.append(f"{i}. {item_text}")
        if len(human_items) > 10:
            lines.append(f"\n... and {len(human_items) - 10} more items")
        lines.append("")

    return "\n".join(lines)


def generate_human_review_report(output: dict) -> str:
    """Generate detailed report of human review items."""
    lines = [
        "# Loop 5 Human Review Items",
        "",
        f"**Run ID:** {output['test_metadata']['run_id']}",
        f"**Topic:** {output['test_metadata']['topic']}",
        f"**Total Items:** {len(output.get('human_review_items', []))}",
        "",
        "---",
        "",
    ]

    human_items = output.get("human_review_items", [])
    for i, item in enumerate(human_items, 1):
        lines.extend([
            f"## Item {i}",
            "",
            str(item),
            "",
        ])

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="Test Loop 5 (Fact & Reference Checking) standalone"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Path to loop 4 output markdown file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--quality",
        choices=VALID_QUALITIES,
        default="test",
        help="Quality level for settings (default: test)",
    )

    args = parser.parse_args()

    try:
        output = await run_loop5_test(
            input_file=args.input_file,
            quality=args.quality,
        )

        timestamp = output["test_metadata"]["timestamp"]
        saved_files = save_test_outputs(output, timestamp)

        print("\n" + "=" * 60)
        print("LOOP 5 TEST COMPLETE")
        print("=" * 60)
        print(f"\nTopic: {output['test_metadata']['topic']}")
        print(f"Duration: {output['test_metadata']['duration_seconds']:.1f}s")
        print(f"Changes: {output['test_metadata']['changes_summary']}")
        print(f"Human Review Items: {output['test_metadata']['human_review_items_count']}")
        print("\nSaved files:")
        for name, path in saved_files.items():
            print(f"  {name}: {path}")

        # Print human review items preview
        human_items = output.get("human_review_items", [])
        if human_items:
            print(f"\nHuman Review Items Preview:")
            for i, item in enumerate(human_items[:5], 1):
                item_preview = str(item)[:100]
                if len(str(item)) > 100:
                    item_preview += "..."
                print(f"  [{i}] {item_preview}")
            if len(human_items) > 5:
                print(f"  ... and {len(human_items) - 5} more items")

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Wait for LangSmith to flush all trace data before exiting
        wait_for_all_tracers()
