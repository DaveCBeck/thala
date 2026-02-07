#!/usr/bin/env python3
"""Run illustrate workflow on deep_dive_3.md."""

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


async def main():
    input_path = Path(__file__).parent.parent / ".context" / "deep_dive_3.md"
    output_path = Path(__file__).parent.parent / ".context" / "artefacts" / "deep_dive_3.md"
    images_dir = output_path.parent / "deep_dive_3_images"

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
        config=IllustrateConfig(
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
        print(f"  - {img.get('location_id')}: {img.get('path')}")


if __name__ == "__main__":
    asyncio.run(main())
