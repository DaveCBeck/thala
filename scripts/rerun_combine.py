#!/usr/bin/env python3
"""Re-run the combine phase on a previously completed web-augmented lit review task.

Two modes:
  --from-checkpoint (default)  Load lit_result + web_result from checkpoint phase_outputs.
  --lit-review PATH --web-research PATH  Load from markdown files (for iterate loops).

Usage:
    .venv/bin/python scripts/rerun_combine.py <task_id_prefix> [--quality quick]
    .venv/bin/python scripts/rerun_combine.py <task_id_prefix> \\
        --lit-review path/to/lit.md --web-research path/to/web.md
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

QUEUE_DIR = PROJECT_ROOT / ".thala" / "queue"
OUTPUT_DIR = PROJECT_ROOT / ".thala" / "output"


def _find_task(prefix: str) -> dict:
    queue_data = json.loads((QUEUE_DIR / "queue.json").read_text())
    for section in ("research_tasks", "publish_tasks"):
        for task in queue_data.get(section, []):
            if task["id"].startswith(prefix):
                return task
    raise SystemExit(f"No task found with prefix '{prefix}'")


def _load_from_checkpoint(task_id: str) -> tuple[dict, dict, str, list[str], str]:
    """Load lit_result and web_result from current_work.json checkpoint.

    Returns (lit_result, web_result, topic, augmented_questions, recent_landscape).
    Raises SystemExit if no matching checkpoint or required data is missing.
    """
    current_work_file = QUEUE_DIR / "current_work.json"
    if not current_work_file.exists():
        raise SystemExit(f"No checkpoint file found at {current_work_file}")

    current_work = json.loads(current_work_file.read_text())
    active_tasks = current_work.get("active_tasks", [])

    for checkpoint in active_tasks:
        if checkpoint.get("task_id", "").startswith(task_id[:8]):
            phase_outputs = checkpoint.get("phase_outputs", {})
            lit_result = phase_outputs.get("lit_result")
            web_result = phase_outputs.get("web_result")
            web_scan_result = phase_outputs.get("web_scan_result", {})

            if not lit_result:
                raise SystemExit(
                    f"Checkpoint for task {task_id[:8]} has no lit_result in phase_outputs.\n"
                    f"Available keys: {list(phase_outputs.keys())}\n"
                    f"Use --lit-review / --web-research to provide files instead."
                )
            if not web_result:
                raise SystemExit(
                    f"Checkpoint for task {task_id[:8]} has no web_result in phase_outputs.\n"
                    f"Available keys: {list(phase_outputs.keys())}\n"
                    f"Use --lit-review / --web-research to provide files instead."
                )

            augmented_questions = web_scan_result.get("augmented_research_questions", [])
            recent_landscape = web_scan_result.get("recent_landscape", "")
            # Derive topic from lit_result or web_scan_result if available
            topic = web_scan_result.get("topic", "")

            return lit_result, web_result, topic, augmented_questions, recent_landscape

    raise SystemExit(
        f"No active checkpoint found for task prefix '{task_id[:8]}'.\n"
        f"Active task IDs: {[c.get('task_id', '')[:8] for c in active_tasks]}\n"
        f"Use --lit-review / --web-research to load from files instead."
    )


def _load_from_files(lit_path: Path, web_path: Path) -> tuple[dict, dict]:
    """Build minimal lit_result and web_result dicts from markdown files."""
    if not lit_path.exists():
        raise SystemExit(f"Lit review file not found: {lit_path}")
    if not web_path.exists():
        raise SystemExit(f"Web research file not found: {web_path}")

    lit_result = {
        "final_report": lit_path.read_text(),
        "paper_corpus": {},
        "paper_summaries": {},
        "zotero_keys": {},
    }
    web_result = {
        "final_report": web_path.read_text(),
        "citation_keys": [],
        "source_count": 0,
    }
    return lit_result, web_result


def _save_output(report: str, topic: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = topic[:50].replace(" ", "_").replace(":", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"combined_{slug}_{timestamp}.md"
    output_path.write_text(report)
    return output_path


async def _run(args: argparse.Namespace) -> None:
    from core.task_queue.workflows.lit_review_web_augmented.phases.combine import run_combine_phase

    task = _find_task(args.task_id_prefix)
    topic = task["topic"]
    research_questions = task.get("research_questions", [])

    print(f"Topic: {topic}")
    print(f"Quality: {args.quality}")

    recent_landscape = ""
    augmented_questions = research_questions

    if args.lit_review and args.web_research:
        # File mode
        print(f"Mode: files ({args.lit_review}, {args.web_research})")
        lit_result, web_result = _load_from_files(Path(args.lit_review), Path(args.web_research))
        if args.augmented_questions:
            augmented_questions = [q.strip() for q in args.augmented_questions.split(",") if q.strip()]
    else:
        # Checkpoint mode
        print("Mode: checkpoint")
        task_id = task["id"]
        lit_result, web_result, checkpoint_topic, checkpoint_questions, recent_landscape = _load_from_checkpoint(
            task_id
        )
        # Checkpoint may have augmented questions; prefer those over task's original
        if checkpoint_questions:
            augmented_questions = checkpoint_questions
        # Override topic from checkpoint scan result if present
        if checkpoint_topic:
            topic = checkpoint_topic

    print(f"Lit report: {len(lit_result.get('final_report', ''))} chars")
    print(f"Web report: {len(web_result.get('final_report', ''))} chars")
    print(f"Research questions: {len(augmented_questions)}")
    print("Running combine phase...")

    combined_result = await run_combine_phase(
        lit_result=lit_result,
        web_result=web_result,
        topic=topic,
        augmented_research_questions=augmented_questions,
        recent_landscape=recent_landscape,
    )

    final_report = combined_result.get("final_report", "")
    if not final_report:
        raise SystemExit("Combine phase returned empty report")

    source_breakdown = combined_result.get("source_breakdown", {})
    print(
        f"Combined: {len(final_report)} chars "
        f"(academic: {source_breakdown.get('academic', 0)}, web: {source_breakdown.get('web', 0)})"
    )

    output_path = _save_output(final_report, topic)
    print(f"Output: {output_path}")


async def main(args: argparse.Namespace) -> None:
    if is_broker_enabled():
        await get_broker().start()

    try:
        await _run(args)
    finally:
        await cleanup_supervisor_resources()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run the combine phase on a web-augmented lit review task"
    )
    parser.add_argument(
        "task_id_prefix",
        help="First 8+ chars of task ID (from queue.json)",
    )
    parser.add_argument(
        "--quality",
        choices=["quick", "standard", "comprehensive", "high_quality"],
        default="quick",
        help="Quality tier (default: quick; passed through for logging only — combine always uses Opus high)",
    )
    parser.add_argument(
        "--lit-review",
        metavar="PATH",
        default=None,
        help="Path to markdown file containing the academic lit review (file mode)",
    )
    parser.add_argument(
        "--web-research",
        metavar="PATH",
        default=None,
        help="Path to markdown file containing the web research report (file mode)",
    )
    parser.add_argument(
        "--augmented-questions",
        metavar="Q1,Q2,...",
        default=None,
        help="Comma-separated augmented research questions (file mode only; defaults to task's research_questions)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    configure_langsmith()
    asyncio.run(main(_parse_args()))
