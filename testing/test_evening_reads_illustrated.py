"""Test script for combined evening_reads + illustrate workflow.

Reads a literature review from a file and transforms it into a 4-part series
(1 overview + 3 deep-dives), then illustrates each article with images.

All outputs are stored under testing/test_data/evening_reads_illustrated_{timestamp}/
"""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.tracers.langchain import wait_for_all_tracers

from testing.utils import configure_logging

configure_logging("evening_reads_illustrated")
logger = logging.getLogger(__name__)


async def run_evening_reads(literature_review: str, output_dir: Path, source_name: str) -> dict:
    """Run the evening_reads workflow on a literature review.

    Args:
        literature_review: The literature review content
        output_dir: Directory to write output files
        source_name: Name of the source file for metadata

    Returns:
        dict with workflow result and list of article paths
    """
    from workflows.output.evening_reads import evening_reads_graph

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
    logger.info(f"Evening reads workflow completed with status: {result.get('status')}")

    if result.get("missing_references"):
        logger.warning(f"Missing references: {result['missing_references']}")

    if result.get("errors"):
        for error in result["errors"]:
            logger.error(f"Error in {error.get('node')}: {error.get('error')}")

    article_paths = []

    # Write final outputs
    final_outputs = result.get("final_outputs", [])
    if final_outputs:
        for output in final_outputs:
            filename = f"{output['id']}.md"
            output_file = output_dir / filename

            header = f"""---
source: {source_name}
generated: {datetime.now().isoformat()}
article_id: {output['id']}
title: {output['title']}
word_count: {output['word_count']}
status: {result.get('status')}
---

# {output['title']}

"""
            output_file.write_text(header + output['content'])
            article_paths.append(output_file)
            logger.info(
                f"Output ({output['id']}): {output['word_count']} words -> {output_file}"
            )
    else:
        logger.error("No final outputs generated")

    # Write planning output
    assignments = result.get("deep_dive_assignments", [])
    overview_scope = result.get("overview_scope", "")
    if assignments:
        plan_file = output_dir / "_plan.md"
        plan_lines = [
            "# Series Plan\n",
            f"**Generated:** {datetime.now().isoformat()}\n",
            f"**Source:** {source_name}\n",
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
        drafts_dir = output_dir / "_drafts"
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
        mappings_file = output_dir / "_citation_mappings.txt"
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
        enriched_file = output_dir / "_enriched_content.txt"
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
        refs_file = output_dir / "_references.md"
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

    # Save generated images and prompts from evening_reads
    image_outputs = result.get("image_outputs", [])
    if image_outputs:
        images_dir = output_dir / "images"
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

    return {
        "result": result,
        "article_paths": article_paths,
        "final_outputs": final_outputs,
    }


async def run_illustrate(input_file: Path, output_dir: Path, article_id: str) -> dict:
    """Run the illustrate workflow on a markdown document.

    Args:
        input_file: Path to the markdown file
        output_dir: Directory to write output files (article-specific subdir)
        article_id: ID of the article being illustrated

    Returns:
        dict with illustration results
    """
    from workflows.output.illustrate import IllustrateConfig, illustrate_graph

    # Read input
    document = input_file.read_text()
    word_count = len(document.split())
    logger.info(f"Illustrating {article_id}: {word_count} words")

    # Create article-specific output directory for images
    images_dir = output_dir / "images" / article_id
    images_dir.mkdir(parents=True, exist_ok=True)

    # Configure workflow
    config = IllustrateConfig(
        generate_header_image=True,
        additional_image_count=2,
        header_prefer_public_domain=True,
        enable_vision_review=True,
        max_retries=1,
        output_dir=str(images_dir),
        enable_diagram_refinement=True,
        diagram_quality_threshold=4.7,
        diagram_max_refinement_iterations=3,
    )

    # Run workflow
    logger.info(f"Starting illustrate workflow for {article_id}...")

    result = await illustrate_graph.ainvoke({
        "input": {
            "markdown_document": document,
            "title": input_file.stem,
            "output_dir": str(images_dir),
        },
        "config": config,
    })

    logger.info(f"Illustrate workflow completed for {article_id} with status: {result.get('status')}")

    if result.get("errors"):
        for error in result["errors"]:
            level = "WARNING" if error.get("severity") == "warning" else "ERROR"
            logger.info(f"  [{level}] {error.get('stage')}: {error.get('message')}")

    # Write illustrated document
    illustrated_doc = result.get("illustrated_document")
    if illustrated_doc:
        output_file = output_dir / f"{article_id}_illustrated.md"
        output_file.write_text(illustrated_doc)
        logger.info(f"Illustrated document written to {output_file}")

    # Write image plan
    image_plan = result.get("image_plan", [])
    if image_plan:
        plan_file = images_dir / "_image_plan.md"
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

    # Write generation results summary
    generation_results = result.get("generation_results", [])
    review_results = result.get("review_results", [])
    final_images = result.get("final_images", [])

    if generation_results or final_images:
        summary_file = images_dir / "_generation_summary.md"
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

    return {
        "result": result,
        "final_images": final_images,
        "image_plan": image_plan,
    }


async def run_combined_workflow(input_file: Path, output_dir: Path) -> Path:
    """Run the complete evening_reads + illustrate workflow.

    Args:
        input_file: Path to the literature review markdown file
        output_dir: Base directory to write output files

    Returns:
        Path to the output directory containing all articles and illustrations
    """
    logger.info(f"Reading literature review from {input_file}")
    literature_review = input_file.read_text()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workflow_dir = output_dir / f"evening_reads_illustrated_{timestamp}"
    workflow_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run evening_reads
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 1: Evening Reads")
    logger.info("=" * 60)

    evening_reads_result = await run_evening_reads(
        literature_review=literature_review,
        output_dir=workflow_dir,
        source_name=input_file.name,
    )

    article_paths = evening_reads_result["article_paths"]
    final_outputs = evening_reads_result["final_outputs"]

    if not article_paths:
        logger.error("No articles generated, cannot proceed with illustration")
        return workflow_dir

    # Step 2: Illustrate each article
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Illustrating Articles")
    logger.info("=" * 60)

    illustration_results = []
    for i, article_path in enumerate(article_paths):
        article_id = article_path.stem
        logger.info("")
        logger.info(f"--- Illustrating article {i+1}/{len(article_paths)}: {article_id} ---")

        try:
            illust_result = await run_illustrate(
                input_file=article_path,
                output_dir=workflow_dir,
                article_id=article_id,
            )
            illustration_results.append({
                "article_id": article_id,
                "success": True,
                "result": illust_result,
            })
        except Exception as e:
            logger.exception(f"Failed to illustrate {article_id}: {e}")
            illustration_results.append({
                "article_id": article_id,
                "success": False,
                "error": str(e),
            })

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Output directory: {workflow_dir}")
    logger.info(f"Articles generated: {len(final_outputs)}")

    total_words = sum(o.get("word_count", 0) for o in final_outputs)
    logger.info(f"Total word count: {total_words}")

    if final_outputs:
        logger.info("")
        logger.info("Articles:")
        for output in final_outputs:
            logger.info(f"  - {output['id']}: \"{output['title']}\" ({output['word_count']} words)")

    successful_illustrations = sum(1 for r in illustration_results if r["success"])
    logger.info("")
    logger.info(f"Articles illustrated: {successful_illustrations}/{len(article_paths)}")

    total_images = 0
    for result in illustration_results:
        if result["success"]:
            total_images += len(result["result"].get("final_images", []))

    logger.info(f"Total images generated: {total_images}")

    if illustration_results:
        logger.info("")
        logger.info("Illustration status:")
        for result in illustration_results:
            status = "SUCCESS" if result["success"] else f"FAILED: {result.get('error', 'Unknown')}"
            image_count = len(result.get("result", {}).get("final_images", [])) if result["success"] else 0
            logger.info(f"  - {result['article_id']}: {status} ({image_count} images)")

    return workflow_dir


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Transform a literature review into an illustrated 4-part series"
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
        output_path = await run_combined_workflow(input_file, output_dir)
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
