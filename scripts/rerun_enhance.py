#!/usr/bin/env python3
"""Re-run the enhancement phase on a previously completed task.

Loads frozen lit_result from checkpoint and calls enhance_report() directly,
bypassing the task queue.

Usage:
    .venv/bin/python scripts/rerun_enhance.py <task_id_prefix> [--quality quick]
    .venv/bin/python scripts/rerun_enhance.py --from-file path/to/review.md --topic "Topic" [--quality quick]
"""

import argparse
import asyncio
import json
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
from workflows.enhance import enhance_report  # noqa: E402

QUEUE_DIR = PROJECT_ROOT / ".thala" / "queue"
OUTPUT_DIR = PROJECT_ROOT / ".thala" / "output"


def _find_task(prefix: str) -> dict:
    queue_data = json.loads((QUEUE_DIR / "queue.json").read_text())
    for section in ("research_tasks", "publish_tasks"):
        for task in queue_data.get(section, []):
            if task["id"].startswith(prefix):
                return task
    raise SystemExit(f"No task found with prefix '{prefix}'")


def _find_checkpoint(task_id: str) -> dict:
    for filename in ("current_work.json", "current_work.previous.json"):
        path = QUEUE_DIR / filename
        if not path.exists():
            continue
        current_work = json.loads(path.read_text())
        for checkpoint in current_work.get("active_tasks", []):
            if checkpoint["task_id"] == task_id:
                return checkpoint
    raise SystemExit(f"No checkpoint found for task_id '{task_id}'")


def _extract_lit_result(checkpoint: dict) -> dict:
    phase_outputs = checkpoint.get("phase_outputs", {})
    task_type = checkpoint.get("task_type", "")
    if task_type == "lit_review_web_augmented":
        lit_result = phase_outputs.get("combined_result") or phase_outputs.get("lit_result")
    else:
        lit_result = phase_outputs.get("lit_result")
    if not lit_result:
        raise SystemExit("No lit_result or combined_result found in checkpoint phase_outputs")
    return lit_result


def _save_output(report: str, topic: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = topic[:50].replace(" ", "_").replace(":", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"lit_review_{slug}_{timestamp}.md"
    output_path.write_text(report)
    return output_path


async def _run_from_checkpoint(task_id_prefix: str, quality: str) -> None:
    task = _find_task(task_id_prefix)
    checkpoint = _find_checkpoint(task["id"])
    lit_result = _extract_lit_result(checkpoint)

    topic = task["topic"]
    research_questions = task.get("research_questions") or lit_result.get("research_questions", [])

    print(f"Topic: {topic}")
    print(f"Quality: {quality}")
    print(f"Task type: {task.get('task_type')}")
    print(f"Report length: {len(lit_result.get('final_report', ''))} chars")
    print("Starting enhancement...")

    result = await enhance_report(
        report=lit_result["final_report"],
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        loops="all",
        paper_corpus=lit_result.get("paper_corpus"),
        paper_summaries=lit_result.get("paper_summaries"),
        zotero_keys=lit_result.get("zotero_keys"),
        run_editing=True,
        run_fact_check=False,
    )

    output_path = _save_output(result["final_report"], topic)
    print(f"Status: {result['status']}")
    if result.get("errors"):
        print(f"Errors: {result['errors']}")
    print(f"Output: {output_path}")


async def _run_from_file(
    file_path: Path, topic: str, quality: str, research_questions: list[str] | None = None
) -> None:
    report = file_path.read_text()

    print(f"Topic: {topic}")
    print(f"Quality: {quality}")
    print(f"File: {file_path}")
    print(f"Report length: {len(report)} chars")
    print("Starting enhancement...")

    result = await enhance_report(
        report=report,
        topic=topic,
        research_questions=research_questions or [],
        quality=quality,
        loops="all",
        paper_corpus=None,
        paper_summaries=None,
        zotero_keys=None,
        run_editing=True,
        run_fact_check=False,
    )

    output_path = _save_output(result["final_report"], topic)
    print(f"Status: {result['status']}")
    if result.get("errors"):
        print(f"Errors: {result['errors']}")
    print(f"Output: {output_path}")


async def main(args: argparse.Namespace) -> None:
    if is_broker_enabled():
        await get_broker().start()

    try:
        if args.from_file:
            await _run_from_file(Path(args.from_file), args.topic, args.quality)
        else:
            await _run_from_checkpoint(args.task_id_prefix, args.quality)
    finally:
        await cleanup_supervisor_resources()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run enhancement phase on a previously completed task"
    )
    parser.add_argument(
        "task_id_prefix",
        nargs="?",
        help="First 8+ chars of task ID (from queue.json)",
    )
    parser.add_argument(
        "--quality",
        choices=["quick", "standard", "comprehensive", "high_quality"],
        default="quick",
        help="Enhancement quality tier (default: quick)",
    )
    parser.add_argument(
        "--from-file",
        metavar="PATH",
        help="Run enhancement on a markdown file instead of loading from checkpoint",
    )
    parser.add_argument(
        "--topic",
        help="Topic string (required with --from-file)",
    )

    args = parser.parse_args()

    if args.from_file:
        if not args.topic:
            parser.error("--topic is required when using --from-file")
        if not Path(args.from_file).exists():
            parser.error(f"File not found: {args.from_file}")
    else:
        if not args.task_id_prefix:
            parser.error("task_id_prefix is required unless --from-file is used")

    return args


if __name__ == "__main__":
    configure_logging()
    configure_langsmith()
    asyncio.run(main(_parse_args()))
