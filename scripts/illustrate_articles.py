#!/usr/bin/env python3
"""Run illustrate workflow on multiple markdown files."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows.output.illustrate import IllustrateConfig, illustrate_document  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Files to process
INPUT_FILES = [
    "deep_dive_1.md",
    "deep_dive_2.md",
    "overview.md",
]


async def process_file(filename: str) -> None:
    """Process a single markdown file through the illustrate workflow."""
    project_root = Path(__file__).parent.parent
    input_path = project_root / ".context" / filename
    stem = input_path.stem
    output_path = project_root / ".context" / "artefacts" / filename
    images_dir = output_path.parent / f"{stem}_images"

    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    # Read input
    markdown = input_path.read_text()

    # Extract title from first line
    title = markdown.split("\n")[0].lstrip("# ").strip()
    print(f"Title: {title}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Run illustrate workflow
    print("Running illustrate workflow...")
    result = await illustrate_document(
        markdown_document=markdown,
        title=title,
        output_dir=str(images_dir),
        options=IllustrateConfig(
            additional_image_count=2,
            enable_vision_review=True,
        ),
    )

    # Write output
    illustrated = result.get("illustrated_document", markdown)
    output_path.write_text(illustrated)
    print(f"Written to: {output_path}")

    # Report results
    final_images = result.get("final_images", [])
    print(f"Images generated: {len(final_images)}")
    for img in final_images:
        print(f"  - {img.get('location_id')}: {img.get('file_path')}")


async def main():
    for filename in INPUT_FILES:
        await process_file(filename)


if __name__ == "__main__":
    asyncio.run(main())
