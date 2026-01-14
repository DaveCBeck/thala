#!/usr/bin/env python3
"""
Download complete LangSmith trace with all child runs and their full inputs/outputs.
Designed for deep analysis of supervision loops.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langsmith import Client


def serialize_datetime(obj):
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def truncate_large_strings(obj, max_len=50000):
    """Truncate very large strings to prevent file bloat while preserving analyzability."""
    if isinstance(obj, str):
        if len(obj) > max_len:
            return obj[:max_len] + f"\n... [TRUNCATED - original length: {len(obj)}]"
        return obj
    elif isinstance(obj, dict):
        return {k: truncate_large_strings(v, max_len) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_large_strings(item, max_len) for item in obj]
    return obj


def download_trace(run_id: str, output_file: str, truncate: bool = True):
    """Download full trace data."""
    client = Client()

    # Get main run
    main_run = client.read_run(run_id)

    print(f"Fetching trace: {main_run.name}")
    print(f"Status: {main_run.status}")
    print(f"Trace ID: {main_run.trace_id}")

    # Get all child runs
    child_runs = list(
        client.list_runs(
            trace_id=str(main_run.trace_id),
        )
    )

    print(f"Found {len(child_runs)} child runs")

    # Build complete trace structure
    trace_data = {
        "meta": {
            "run_id": run_id,
            "trace_id": str(main_run.trace_id),
            "name": main_run.name,
            "status": main_run.status,
            "start_time": main_run.start_time,
            "end_time": main_run.end_time,
            "total_tokens": main_run.total_tokens,
            "prompt_tokens": main_run.prompt_tokens,
            "completion_tokens": main_run.completion_tokens,
            "child_run_count": len(child_runs),
        },
        "main_run": {
            "inputs": main_run.inputs,
            "outputs": main_run.outputs,
        },
        "child_runs": [],
    }

    # Process child runs - organize by name for easier analysis
    for run in sorted(child_runs, key=lambda r: r.start_time or datetime.min):
        run_data = {
            "id": str(run.id),
            "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
            "name": run.name,
            "run_type": run.run_type,
            "status": run.status,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "total_tokens": run.total_tokens,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "inputs": run.inputs,
            "outputs": run.outputs,
            "error": run.error,
        }

        if truncate:
            run_data = truncate_large_strings(run_data)

        trace_data["child_runs"].append(run_data)

    # Write to file
    with open(output_file, "w") as f:
        json.dump(trace_data, f, indent=2, default=serialize_datetime)

    print(f"Wrote trace to {output_file}")

    # Print summary of run types
    run_types = {}
    for run in child_runs:
        name = run.name or "unnamed"
        run_types[name] = run_types.get(name, 0) + 1

    print("\nRun type summary:")
    for name, count in sorted(run_types.items(), key=lambda x: -x[1])[:30]:
        print(f"  {name}: {count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_full_trace.py <run_id> [output_file]")
        sys.exit(1)

    run_id = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"trace_{run_id}.json"

    download_trace(run_id, output_file)
