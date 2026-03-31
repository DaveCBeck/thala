#!/usr/bin/env python3
"""Re-run evening reads (overview + 3 deep-dives) from a literature review file.

Loads a literature review markdown file, applies an editorial stance,
and runs the evening_reads workflow to produce 4 articles.

Usage:
    .venv/bin/python scripts/rerun_evening_reads.py \
        --from-file .thala/output/lit_review_Example_20260329.md \
        --publication reasoning-under-uncertainty
"""

import argparse
import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from core.config import configure_logging, configure_langsmith  # noqa: E402
from core.llm_broker import get_broker, is_broker_enabled  # noqa: E402
from core.task_queue.lifecycle import cleanup_supervisor_resources  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / ".thala" / "output"


def _save_outputs(final_outputs: list[dict], topic: str, publication: str) -> Path:
    """Save each article to a timestamped directory."""
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", topic[:50]).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"evening_reads_{slug}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for article in final_outputs:
        article_id = article["id"]
        title = article.get("title", article_id)
        subtitle = article.get("subtitle", "")
        content = article.get("content", "")
        word_count = article.get("word_count", 0)

        header = f"# {title}\n"
        if subtitle:
            header += f"\n*{subtitle}*\n"
        header += f"\n*{word_count} words | Publication: {publication}*\n\n---\n\n"

        (output_dir / f"{article_id}.md").write_text(header + content)

    return output_dir


async def _run(from_file: str, publication: str) -> None:
    from workflows.output.evening_reads import evening_reads
    from workflows.output.evening_reads.editorial import load_editorial_stance_by_slug

    lit_review_path = Path(from_file)
    if not lit_review_path.exists():
        raise SystemExit(f"File not found: {lit_review_path}")

    literature_review = lit_review_path.read_text()
    print(f"Loaded lit review: {len(literature_review)} chars")

    editorial_stance = load_editorial_stance_by_slug(publication) or ""
    if editorial_stance:
        print(f"Using editorial stance: {publication}")
    else:
        print(f"No editorial stance found for: {publication}")

    topic = "Untitled"
    for line in literature_review.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            topic = re.sub(r"^#+\s*", "", stripped).strip()
            break

    print(f"Topic: {topic}")
    print("Running evening reads workflow...")

    result = await evening_reads(literature_review, editorial_stance)

    final_outputs = result.get("final_outputs", [])
    status = result.get("status", "unknown")
    errors = result.get("errors", [])

    print(f"Status: {status}")
    print(f"Articles: {len(final_outputs)}")

    if errors:
        for err in errors:
            print(f"  Error: {err}")

    if not final_outputs:
        raise SystemExit("Evening reads produced no output")

    output_dir = _save_outputs(final_outputs, topic, publication)
    print(f"Output: {output_dir}")

    for article in final_outputs:
        print(f"  {article['id']}: {article.get('title', 'Untitled')} ({article.get('word_count', 0)} words)")


async def main(args: argparse.Namespace) -> None:
    if is_broker_enabled():
        await get_broker().start()

    try:
        await _run(args.from_file, args.publication)
    finally:
        await cleanup_supervisor_resources()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run evening reads from a literature review file",
    )
    parser.add_argument(
        "--from-file",
        required=True,
        help="Path to literature review markdown file",
    )
    parser.add_argument(
        "--publication",
        required=True,
        help="Publication slug for editorial stance (e.g., reasoning-under-uncertainty)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    configure_langsmith()
    asyncio.run(main(_parse_args()))
