#!/usr/bin/env python3
"""Re-run web research using the agent-team workflow.

Usage:
    .venv/bin/python scripts/rerun_web_research_team.py <task_id_prefix> [--quality quick] [--window-days 30]
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from core.config import configure_logging, configure_langsmith  # noqa: E402

QUEUE_DIR = PROJECT_ROOT / ".thala" / "queue"
OUTPUT_DIR = PROJECT_ROOT / ".thala" / "output"


def _find_task(prefix: str) -> dict:
    queue_data = json.loads((QUEUE_DIR / "queue.json").read_text())
    for section in ("research_tasks", "publish_tasks"):
        for task in queue_data.get(section, []):
            if task["id"].startswith(prefix):
                return task
    raise SystemExit(f"No task found with prefix '{prefix}'")


def _build_recency_filter(window_days: int) -> dict:
    after_date = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%d")
    return {"after_date": after_date, "quota": 0.3}


def _save_output(report: str, topic: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = topic[:50].replace(" ", "_").replace(":", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"web_research_team_{slug}_{timestamp}.md"
    output_path.write_text(report)
    return output_path


async def _run(task_id_prefix: str, quality: str, window_days: int) -> None:
    from workflows.research.web_research_team.api import deep_research_team

    task = _find_task(task_id_prefix)
    topic = task["topic"]
    recency_filter = _build_recency_filter(window_days)

    print(f"Topic: {topic}")
    print(f"Quality: {quality}")
    print(f"Window days: {window_days} (after {recency_filter['after_date']})")
    print("Running agent-team research...")

    result = await deep_research_team(
        query=topic,
        quality=quality,
        recency_filter=recency_filter,
    )

    final_report = result.get("final_report", "")
    status = result.get("status", "unknown")
    source_count = result.get("source_count", 0)
    errors = result.get("errors", [])

    print(f"Status: {status}")
    print(f"Sources: {source_count}")
    if errors:
        print(f"Errors: {len(errors)}")

    if not final_report:
        raise SystemExit(f"Research produced no report (status={status})")

    print(f"Report: {len(final_report)} chars")

    output_path = _save_output(final_report, topic)
    print(f"Output: {output_path}")


async def main(args: argparse.Namespace) -> None:
    await _run(args.task_id_prefix, args.quality, args.window_days)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run web research using agent-team workflow"
    )
    parser.add_argument("task_id_prefix", help="First 8+ chars of task ID")
    parser.add_argument(
        "--quality",
        choices=["test", "quick", "standard", "comprehensive", "high_quality"],
        default="quick",
    )
    parser.add_argument("--window-days", type=int, default=30, dest="window_days")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    configure_langsmith()
    asyncio.run(main(_parse_args()))
