#!/usr/bin/env python3
"""Re-run clustering + synthesis (+ optional enhancement) on a previously completed task.

Loads frozen paper_summaries/zotero_keys from the saved workflow state
and re-runs clustering → synthesis → (optionally) enhancement.

Usage:
    # From task ID (looks up saved workflow state by matching topic)
    .venv/bin/python scripts/rerun_synthesis.py <task_id_prefix> [--quality quick] [--enhance]

    # --enhance: also run supervision + editing after synthesis
"""

import argparse
import asyncio
import json
import gzip
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
STATE_STORE_DIR = Path.home() / ".thala" / "workflow_states" / "academic_lit_review"


def _find_task(prefix: str) -> dict:
    queue_data = json.loads((QUEUE_DIR / "queue.json").read_text())
    for section in ("research_tasks", "publish_tasks"):
        for task in queue_data.get(section, []):
            if task["id"].startswith(prefix):
                return task
    raise SystemExit(f"No task found with prefix '{prefix}'")


def _load_workflow_state(topic: str) -> dict:
    """Find the most recent workflow state matching this topic."""
    if not STATE_STORE_DIR.exists():
        raise SystemExit(f"No workflow states directory: {STATE_STORE_DIR}")

    # Sort by modification time, newest first
    state_files = sorted(STATE_STORE_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

    for path in state_files:
        try:
            if path.suffix == ".gz":
                data = json.loads(gzip.open(path).read())
            else:
                data = json.loads(path.read_text())

            state_topic = data.get("input", {}).get("topic", "")
            if state_topic == topic:
                return data
        except (json.JSONDecodeError, OSError):
            continue

    raise SystemExit(
        f"No saved workflow state found for topic '{topic[:60]}...'\n"
        f"Searched {len(state_files)} files in {STATE_STORE_DIR}"
    )


def _save_output(report: str, topic: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = topic[:50].replace(" ", "_").replace(":", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"lit_review_{slug}_{timestamp}.md"
    output_path.write_text(report)
    return output_path


async def _run(task_id_prefix: str, quality: str, enhance: bool, category_override: str | None = None) -> None:
    from workflows.research.academic_lit_review.clustering.api import run_clustering
    from workflows.research.academic_lit_review.synthesis.api import run_synthesis
    from workflows.research.academic_lit_review.quality_presets import QUALITY_PRESETS

    task = _find_task(task_id_prefix)
    topic = task["topic"]
    research_questions = task.get("research_questions", [])

    # Load editorial stance from task category (or --category override)
    category = category_override or task.get("category")
    editorial_stance = None
    if category:
        from workflows.output.evening_reads.editorial import load_editorial_stance

        editorial_stance = load_editorial_stance(category)

    print(f"Topic: {topic}")
    print(f"Quality: {quality}")
    print(f"Enhance: {enhance}")
    print(f"Category: {category or '(none)'}")
    print(f"Editorial stance: {'loaded' if editorial_stance else 'none'}")

    # Load frozen state
    state = _load_workflow_state(topic)
    paper_summaries = state["paper_summaries"]
    zotero_keys = state["zotero_keys"]
    quality_settings = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["quick"])

    if not research_questions:
        research_questions = state.get("input", {}).get("research_questions", [])

    print(f"Papers: {len(paper_summaries)}")
    print(f"Zotero keys: {len(zotero_keys)}")
    print("Running clustering...")

    # Step 1: Clustering
    clustering_result = await run_clustering(
        paper_summaries=paper_summaries,
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
    )

    clusters = clustering_result.get("final_clusters", [])
    cluster_analyses = clustering_result.get("cluster_analyses", [])
    print(f"Clusters: {len(clusters)}")

    if not clusters:
        raise SystemExit("Clustering produced no clusters — cannot synthesize")

    # Step 2: Synthesis
    print("Running synthesis...")
    synthesis_result = await run_synthesis(
        paper_summaries=paper_summaries,
        clusters=clusters,
        cluster_analyses=cluster_analyses,
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
        zotero_keys=zotero_keys,
        editorial_stance=editorial_stance,
    )

    final_report = synthesis_result.get("final_review", "")
    print(f"Synthesis complete: {len(final_report)} chars")

    # Step 3: Optional enhancement
    if enhance and final_report:
        from workflows.enhance import enhance_report

        print("Running enhancement...")
        enhance_result = await enhance_report(
            report=final_report,
            topic=topic,
            research_questions=research_questions,
            quality=quality,
            loops="all",
            paper_corpus=state.get("paper_corpus"),
            paper_summaries=paper_summaries,
            zotero_keys=zotero_keys,
            run_editing=True,
            run_fact_check=False,
        )
        final_report = enhance_result["final_report"]
        print(f"Enhancement: status={enhance_result['status']}, {len(final_report)} chars")

    output_path = _save_output(final_report, topic)
    print(f"Output: {output_path}")


async def main(args: argparse.Namespace) -> None:
    if is_broker_enabled():
        await get_broker().start()

    try:
        await _run(args.task_id_prefix, args.quality, args.enhance, args.category)
    finally:
        await cleanup_supervisor_resources()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run clustering + synthesis on a previously completed task"
    )
    parser.add_argument(
        "task_id_prefix",
        help="First 8+ chars of task ID (from queue.json)",
    )
    parser.add_argument(
        "--quality",
        choices=["quick", "standard", "comprehensive", "high_quality"],
        default="quick",
        help="Quality tier (default: quick)",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Also run supervision + editing after synthesis",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Publication category override (default: from task in queue.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    configure_langsmith()
    asyncio.run(main(_parse_args()))
