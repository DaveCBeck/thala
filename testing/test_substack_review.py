"""Test script for substack_review workflow.

Reads a literature review from a file and transforms it into a Substack essay.
"""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.tracers.langchain import wait_for_all_tracers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_substack_review(input_file: Path, output_dir: Path) -> Path:
    """Run the substack review workflow on a literature review file.

    Args:
        input_file: Path to the literature review markdown file
        output_dir: Directory to write output files

    Returns:
        Path to the output file
    """
    from workflows.output.substack_review import substack_review_graph

    # Read input
    logger.info(f"Reading literature review from {input_file}")
    literature_review = input_file.read_text()
    word_count = len(literature_review.split())
    logger.info(f"Input: {word_count} words")

    # Run workflow
    logger.info("Starting substack_review workflow...")
    logger.info(
        "This will generate 3 essays in parallel with OPUS, then select the best one"
    )

    result = await substack_review_graph.ainvoke(
        {"input": {"literature_review": literature_review}}
    )

    # Log results
    logger.info(f"Workflow completed with status: {result.get('status')}")
    logger.info(f"Selected angle: {result.get('selected_angle')}")
    logger.info(f"Selection reasoning: {result.get('selection_reasoning')}")

    if result.get("missing_references"):
        logger.warning(f"Missing references: {result['missing_references']}")

    if result.get("errors"):
        for error in result["errors"]:
            logger.error(f"Error in {error.get('node')}: {error.get('error')}")

    # Write output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"substack_essay_{timestamp}.md"

    final_essay = result.get("final_essay", "")
    if final_essay:
        # Add metadata header
        header = f"""---
source: {input_file.name}
generated: {datetime.now().isoformat()}
selected_angle: {result.get("selected_angle")}
status: {result.get("status")}
---

"""
        output_file.write_text(header + final_essay)
        logger.info(f"Output written to {output_file}")

        output_word_count = len(final_essay.split())
        logger.info(f"Output: {output_word_count} words")
    else:
        logger.error("No final essay generated")
        # Write debug info
        debug_file = output_dir / f"substack_debug_{timestamp}.txt"
        debug_file.write_text(str(result))
        logger.info(f"Debug info written to {debug_file}")

    # Also write all drafts for comparison
    essay_drafts = result.get("essay_drafts", [])
    if essay_drafts:
        drafts_dir = output_dir / f"drafts_{timestamp}"
        drafts_dir.mkdir(exist_ok=True)
        for draft in essay_drafts:
            draft_file = drafts_dir / f"{draft['angle']}_essay.md"
            draft_file.write_text(draft["content"])
            logger.info(
                f"Draft ({draft['angle']}): {draft['word_count']} words -> {draft_file}"
            )

    # Write evaluations
    evaluations = result.get("essay_evaluations")
    if evaluations:
        eval_file = output_dir / f"evaluations_{timestamp}.txt"
        eval_lines = []
        for angle, eval_data in evaluations.items():
            eval_lines.append(f"\n=== {angle.upper()} ===")
            eval_lines.append(f"Primary strength: {eval_data.get('primary_strength')}")
            eval_lines.append(f"Primary weakness: {eval_data.get('primary_weakness')}")
            eval_lines.append(
                f"Scores: hook={eval_data.get('hook_strength')}, "
                f"momentum={eval_data.get('structural_momentum')}, "
                f"payoff={eval_data.get('technical_payoff')}, "
                f"tone={eval_data.get('tonal_calibration')}, "
                f"complexity={eval_data.get('honest_complexity')}, "
                f"fit={eval_data.get('subject_fit')}"
            )
        eval_file.write_text("\n".join(eval_lines))
        logger.info(f"Evaluations written to {eval_file}")

    return output_file


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Transform a literature review into a Substack essay"
    )
    parser.add_argument("input_file", type=Path, help="Path to the literature review markdown file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/dave/thala/testing/test_data"),
        help="Directory to write output files (default: testing/test_data)",
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        output_file = await run_substack_review(input_file, output_dir)
        logger.info(f"Test complete! Output: {output_file}")
    except Exception as e:
        logger.exception(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Wait for LangSmith to flush all trace data before exiting
        wait_for_all_tracers()
