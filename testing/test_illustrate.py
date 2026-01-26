"""Test script for illustrate workflow.

Takes a markdown document and adds images using three sources:
- Public domain images (Pexels/Unsplash)
- AI-generated images (Imagen)
- SVG diagrams (Claude-generated)

Usage:
    python testing/test_illustrate.py testing/test_data/evening_reads_20260126_121649/deep_dive_3.md
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


async def run_illustrate(input_file: Path, output_dir: Path) -> Path:
    """Run the illustrate workflow on a markdown document.

    Args:
        input_file: Path to the markdown file
        output_dir: Directory to write output files

    Returns:
        Path to the output directory containing images and illustrated document
    """
    from workflows.output.illustrate import IllustrateConfig, illustrate_graph

    # Read input
    logger.info(f"Reading document from {input_file}")
    document = input_file.read_text()
    word_count = len(document.split())
    logger.info(f"Input: {word_count} words")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    illustrate_dir = output_dir / f"illustrate_{timestamp}"
    illustrate_dir.mkdir(parents=True, exist_ok=True)

    # Configure workflow
    config = IllustrateConfig(
        generate_header_image=True,
        additional_image_count=2,
        header_prefer_public_domain=True,
        enable_vision_review=True,
        max_retries=1,
        output_dir=str(illustrate_dir / "images"),
        # Diagram quality refinement settings
        enable_diagram_refinement=True,
        diagram_quality_threshold=4.7,
        diagram_max_refinement_iterations=3,
    )

    # Run workflow
    logger.info("Starting illustrate workflow...")
    logger.info(
        f"Config: header={config.generate_header_image}, "
        f"additional={config.additional_image_count}, "
        f"pd_prefer={config.header_prefer_public_domain}, "
        f"review={config.enable_vision_review}"
    )

    result = await illustrate_graph.ainvoke({
        "input": {
            "markdown_document": document,
            "title": input_file.stem,
            "output_dir": str(illustrate_dir / "images"),
        },
        "config": config,
    })

    # Log results
    logger.info(f"Workflow completed with status: {result.get('status')}")

    if result.get("errors"):
        logger.info("")
        logger.info("Errors/Warnings:")
        for error in result["errors"]:
            level = "WARNING" if error.get("severity") == "warning" else "ERROR"
            logger.info(f"  [{level}] {error.get('stage')}: {error.get('message')}")

    # Write illustrated document
    illustrated_doc = result.get("illustrated_document")
    if illustrated_doc:
        output_file = illustrate_dir / f"{input_file.stem}_illustrated.md"
        output_file.write_text(illustrated_doc)
        logger.info(f"Illustrated document written to {output_file}")

    # Write image plan
    image_plan = result.get("image_plan", [])
    if image_plan:
        plan_file = illustrate_dir / "_image_plan.md"
        plan_lines = [
            "# Image Plan",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Source:** {input_file.name}",
            "",
        ]
        for plan in image_plan:
            plan_lines.append(f"## {plan.location_id}: {plan.purpose}")
            plan_lines.append("")
            plan_lines.append(f"**Insert after:** {plan.insertion_after_header}")
            plan_lines.append(f"**Type:** {plan.image_type}")
            plan_lines.append(f"**Rationale:** {plan.type_rationale}")
            plan_lines.append("")
            plan_lines.append("**Brief:**")
            plan_lines.append("```")
            plan_lines.append(plan.brief[:500] + ("..." if len(plan.brief) > 500 else ""))
            plan_lines.append("```")
            plan_lines.append("")

        plan_file.write_text("\n".join(plan_lines))
        logger.info(f"Image plan written to {plan_file}")

    # Write generation results summary
    generation_results = result.get("generation_results", [])
    review_results = result.get("review_results", [])
    final_images = result.get("final_images", [])

    if generation_results or final_images:
        summary_file = illustrate_dir / "_generation_summary.md"
        summary_lines = [
            "# Generation Summary",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            "",
            "## Generation Results",
            "",
        ]

        for gen in generation_results:
            status = "SUCCESS" if gen.get("success") else "FAILED"
            summary_lines.append(f"### {gen.get('location_id')}: {status}")
            summary_lines.append("")
            summary_lines.append(f"- **Type:** {gen.get('image_type')}")
            summary_lines.append(f"- **Prompt/Query:** {gen.get('prompt_or_query_used', '')[:100]}...")
            if gen.get("attribution"):
                attr = gen["attribution"]
                summary_lines.append(f"- **Attribution:** {attr.get('photographer')} via {attr.get('source')}")
            summary_lines.append("")

        if review_results:
            summary_lines.append("## Review Results")
            summary_lines.append("")
            for review in review_results:
                status = "PASSED" if review.get("passed") else "FAILED"
                summary_lines.append(f"### {review.get('location_id')}: {status}")
                if review.get("severity"):
                    summary_lines.append(f"- **Severity:** {review['severity']}")
                if review.get("issues"):
                    summary_lines.append(f"- **Issues:** {', '.join(review['issues'][:3])}")
                summary_lines.append("")

        if final_images:
            summary_lines.append("## Final Images")
            summary_lines.append("")
            for img in final_images:
                summary_lines.append(f"- **{img['location_id']}:** {img['file_path']}")
                summary_lines.append(f"  - Alt: {img['alt_text']}")
                if img.get("attribution"):
                    attr = img["attribution"]
                    req = "Required" if attr.get("required") else "Optional"
                    summary_lines.append(f"  - Attribution ({req}): {attr.get('photographer')}")
            summary_lines.append("")

        summary_file.write_text("\n".join(summary_lines))
        logger.info(f"Generation summary written to {summary_file}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Output directory: {illustrate_dir}")
    logger.info(f"Status: {result.get('status')}")
    logger.info(f"Images planned: {len(image_plan)}")
    logger.info(f"Images generated: {sum(1 for g in generation_results if g.get('success'))}")
    logger.info(f"Images finalized: {len(final_images)}")

    if final_images:
        logger.info("")
        logger.info("Final images:")
        for img in final_images:
            logger.info(f"  - {img['location_id']}: {Path(img['file_path']).name}")

    warning_count = sum(1 for e in result.get("errors", []) if e.get("severity") == "warning")
    error_count = sum(1 for e in result.get("errors", []) if e.get("severity") == "error")
    if warning_count or error_count:
        logger.info(f"Warnings: {warning_count}, Errors: {error_count}")

    return illustrate_dir


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Add images to a markdown document using AI"
    )
    parser.add_argument(
        "input_file", type=Path, help="Path to the markdown file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/dave/thala/testing/test_data"),
        help="Directory to write output files (default: testing/test_data)",
    )
    parser.add_argument(
        "--additional-images",
        type=int,
        default=2,
        help="Number of additional images beyond header (default: 2)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Skip header image generation",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip vision review of generated images",
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        output_path = await run_illustrate(input_file, output_dir)
        logger.info(f"Test complete! Output: {output_path}")
    except Exception as e:
        logger.exception(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Wait for LangSmith to flush all trace data before exiting
        wait_for_all_tracers()
