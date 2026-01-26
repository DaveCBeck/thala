"""Test script for evening_reads workflow.

Reads a literature review from a file and transforms it into a 4-part series:
1 overview + 3 deep-dives.
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


async def run_evening_reads(input_file: Path, output_dir: Path) -> Path:
    """Run the evening_reads workflow on a literature review file.

    Args:
        input_file: Path to the literature review markdown file
        output_dir: Directory to write output files

    Returns:
        Path to the output directory containing all articles
    """
    from workflows.output.evening_reads import evening_reads_graph

    # Read input
    logger.info(f"Reading literature review from {input_file}")
    literature_review = input_file.read_text()
    word_count = len(literature_review.split())
    logger.info(f"Input: {word_count} words")

    # Run workflow
    logger.info("Starting evening_reads workflow...")
    logger.info(
        "This will plan 3 deep-dive topics, fetch content, then write 4 articles in parallel with OPUS"
    )

    result = await evening_reads_graph.ainvoke(
        {"input": {"literature_review": literature_review}}
    )

    # Log results
    logger.info(f"Workflow completed with status: {result.get('status')}")

    if result.get("missing_references"):
        logger.warning(f"Missing references: {result['missing_references']}")

    if result.get("errors"):
        for error in result["errors"]:
            logger.error(f"Error in {error.get('node')}: {error.get('error')}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    series_dir = output_dir / f"evening_reads_{timestamp}"
    series_dir.mkdir(parents=True, exist_ok=True)

    # Write final outputs
    final_outputs = result.get("final_outputs", [])
    if final_outputs:
        for output in final_outputs:
            # Create filename from id
            filename = f"{output['id']}.md"
            output_file = series_dir / filename

            # Add metadata header
            header = f"""---
source: {input_file.name}
generated: {datetime.now().isoformat()}
article_id: {output['id']}
title: {output['title']}
word_count: {output['word_count']}
status: {result.get('status')}
---

# {output['title']}

"""
            output_file.write_text(header + output['content'])
            logger.info(
                f"Output ({output['id']}): {output['word_count']} words -> {output_file}"
            )
    else:
        logger.error("No final outputs generated")

    # Write planning output
    assignments = result.get("deep_dive_assignments", [])
    overview_scope = result.get("overview_scope", "")
    if assignments:
        plan_file = series_dir / "_plan.md"
        plan_lines = [
            "# Series Plan\n",
            f"**Generated:** {datetime.now().isoformat()}\n",
            f"**Source:** {input_file.name}\n",
            "",
            "## Overview Scope",
            "",
            overview_scope,
            "",
            "## Deep-Dive Assignments",
            "",
        ]
        for assignment in assignments:
            plan_lines.append(f"### {assignment['id']}: {assignment['title']}")
            plan_lines.append("")
            plan_lines.append(f"**Theme:** {assignment['theme']}")
            plan_lines.append("")
            plan_lines.append(f"**Structural Approach:** {assignment.get('structural_approach', 'not specified')}")
            plan_lines.append("")
            plan_lines.append(f"**Anchor Keys:** {', '.join(assignment['anchor_keys'])}")
            plan_lines.append("")
            plan_lines.append(f"**Relevant Sections:** {', '.join(assignment['relevant_sections'])}")
            plan_lines.append("")

        plan_file.write_text("\n".join(plan_lines))
        logger.info(f"Plan written to {plan_file}")

    # Write draft versions (before reference formatting)
    deep_dive_drafts = result.get("deep_dive_drafts", [])
    overview_draft = result.get("overview_draft")

    if deep_dive_drafts or overview_draft:
        drafts_dir = series_dir / "_drafts"
        drafts_dir.mkdir(exist_ok=True)

        for draft in deep_dive_drafts:
            draft_file = drafts_dir / f"{draft['id']}_draft.md"
            draft_file.write_text(f"# {draft['title']}\n\n{draft['content']}")
            logger.info(
                f"Draft ({draft['id']}): {draft['word_count']} words -> {draft_file}"
            )

        if overview_draft:
            draft_file = drafts_dir / "overview_draft.md"
            draft_file.write_text(
                f"# {overview_draft['title']}\n\n{overview_draft['content']}"
            )
            logger.info(
                f"Draft (overview): {overview_draft['word_count']} words -> {draft_file}"
            )

    # Write citation mappings
    citation_mappings = result.get("citation_mappings", {})
    if citation_mappings:
        mappings_file = series_dir / "_citation_mappings.txt"
        mapping_lines = ["Citation Key Mappings", "=" * 40, ""]
        for key, mapping in sorted(citation_mappings.items()):
            es_id = mapping.get("es_record_id", "NOT FOUND")
            title = mapping.get("title", "Unknown")
            mapping_lines.append(f"[@{key}]")
            mapping_lines.append(f"  ES Record: {es_id}")
            mapping_lines.append(f"  Title: {title}")
            mapping_lines.append("")
        mappings_file.write_text("\n".join(mapping_lines))
        logger.info(f"Citation mappings written to {mappings_file}")

    # Write enriched content summary
    enriched_content = result.get("enriched_content", [])
    if enriched_content:
        enriched_file = series_dir / "_enriched_content.txt"
        enriched_lines = ["Enriched Content Summary", "=" * 40, ""]
        for ec in enriched_content:
            enriched_lines.append(f"Deep-dive: {ec['deep_dive_id']}")
            enriched_lines.append(f"  Key: {ec['zotero_key']}")
            enriched_lines.append(f"  Level: {ec['content_level']}")
            enriched_lines.append(f"  Length: {len(ec['content'])} chars")
            enriched_lines.append("")
        enriched_file.write_text("\n".join(enriched_lines))
        logger.info(f"Enriched content summary written to {enriched_file}")

    # Write formatted references
    formatted_refs = result.get("formatted_references", [])
    if formatted_refs:
        refs_file = series_dir / "_references.md"
        ref_lines = ["# All References", ""]
        found_refs = [r for r in formatted_refs if r.get("found_in_zotero")]
        missing_refs = [r for r in formatted_refs if not r.get("found_in_zotero")]

        if found_refs:
            ref_lines.append("## Found in Zotero")
            ref_lines.append("")
            for ref in sorted(found_refs, key=lambda r: r["citation_text"]):
                ref_lines.append(f"- [@{ref['key']}] {ref['citation_text']}")
            ref_lines.append("")

        if missing_refs:
            ref_lines.append("## Not Found")
            ref_lines.append("")
            for ref in missing_refs:
                ref_lines.append(f"- [@{ref['key']}] {ref['citation_text']}")
            ref_lines.append("")

        refs_file.write_text("\n".join(ref_lines))
        logger.info(f"References written to {refs_file}")

    # Save generated images and prompts
    image_outputs = result.get("image_outputs", [])
    if image_outputs:
        images_dir = series_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for img_output in image_outputs:
            article_id = img_output["article_id"]
            image_bytes = img_output["image_bytes"]
            prompt_used = img_output["prompt_used"]

            # Save image
            image_file = images_dir / f"{article_id}.png"
            image_file.write_bytes(image_bytes)

            # Save prompt used for this image
            prompt_file = images_dir / f"{article_id}_prompt.txt"
            prompt_file.write_text(prompt_used)

            logger.info(
                f"Image ({article_id}): {len(image_bytes)} bytes -> {image_file}"
            )

        # Write image manifest
        manifest_file = images_dir / "_manifest.txt"
        manifest_lines = ["Generated Images", "=" * 40, ""]
        for img_output in image_outputs:
            manifest_lines.append(f"Article: {img_output['article_id']}")
            manifest_lines.append(f"  Image: {img_output['article_id']}.png")
            manifest_lines.append(f"  Prompt: {img_output['article_id']}_prompt.txt")
            manifest_lines.append(f"  Size: {len(img_output['image_bytes'])} bytes")
            manifest_lines.append("")
        manifest_file.write_text("\n".join(manifest_lines))
        logger.info(f"Image manifest written to {manifest_file}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Output directory: {series_dir}")
    logger.info(f"Status: {result.get('status')}")
    logger.info(f"Articles generated: {len(final_outputs)}")
    logger.info(f"Images generated: {len(image_outputs)}")

    total_words = sum(o.get("word_count", 0) for o in final_outputs)
    logger.info(f"Total word count: {total_words}")

    if final_outputs:
        logger.info("")
        logger.info("Articles:")
        for output in final_outputs:
            logger.info(f"  - {output['id']}: \"{output['title']}\" ({output['word_count']} words)")

    return series_dir


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Transform a literature review into a 4-part series"
    )
    parser.add_argument(
        "input_file", type=Path, help="Path to the literature review markdown file"
    )
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
        output_path = await run_evening_reads(input_file, output_dir)
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
