#!/usr/bin/env python3
"""One-shot backfill: export existing lit reviews to Quartz content directory.

Reads lit_review_*.md and matching summary_*.json from a given output directory,
strips the metadata header (above the --- separator), and exports via the same
quartz_export logic used by save_and_spawn.

Usage:
    python scripts/backfill_quartz_reviews.py [output_dir]

    output_dir defaults to .worktrees/live-running/.thala/output/
"""

import asyncio
import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.task_queue.workflows.shared.quartz_export import export_lit_review_to_quartz


def _strip_metadata_header(text: str) -> str:
    """Strip the metadata block above the --- separator.

    The lit_review_*.md files have a metadata preamble:
        # Literature Review: Topic
        *Generated: ...*
        *Quality: ...*
        ## Research Questions
        - ...
        ---
        <actual content starts here>

    Returns the content below the --- separator.
    """
    parts = text.split("\n---\n", maxsplit=1)
    if len(parts) == 2:
        return parts[1].lstrip("\n")
    return text


async def backfill(output_dir: Path) -> None:
    lit_reviews = sorted(output_dir.glob("lit_review_*.md"))
    print(f"Found {len(lit_reviews)} lit review files in {output_dir}")

    for lr_path in lit_reviews:
        # Find matching summary JSON (same suffix after lit_review_ / summary_)
        suffix = lr_path.name.removeprefix("lit_review_")
        summary_path = output_dir / f"summary_{suffix.removesuffix('.md')}.json"

        if not summary_path.exists():
            print(f"  SKIP {lr_path.name}: no matching summary JSON")
            continue

        with open(summary_path) as f:
            meta = json.load(f)

        raw_content = lr_path.read_text()
        content = _strip_metadata_header(raw_content)

        result = await export_lit_review_to_quartz(
            content=content,
            topic=meta["topic"],
            category=meta["category"],
            generated_at=meta["generated_at"],
            quality=meta["quality"],
        )

        if result:
            print(f"  OK {result.relative_to(result.parent.parent.parent)}")
        else:
            print(f"  FAIL {lr_path.name}: export returned None")


def main() -> None:
    default_dir = Path(".worktrees/live-running/.thala/output")
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir

    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist")
        sys.exit(1)

    asyncio.run(backfill(output_dir))
    print("\nDone.")


if __name__ == "__main__":
    main()
