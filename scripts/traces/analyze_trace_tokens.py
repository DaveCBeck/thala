#!/usr/bin/env python3
"""
Analyze token usage per function/node for a LangSmith trace.

Usage:
    python analyze_trace_tokens.py <trace_id_or_run_id>
    python analyze_trace_tokens.py 019bf08c-460d-7b92-bd5b-2d9a9846caf7

Note: Due to a LangSmith API bug, list_runs(trace_id=...) returns 0 for token fields.
This script works around it by querying run_ids in batches after getting IDs.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from langsmith import Client


def analyze_trace(trace_id: str, batch_size: int = 100) -> dict:
    """Fetch and analyze token usage for a trace."""
    client = Client()

    # First, read the root run to get the correct trace_id
    print(f"Reading root run {trace_id}...")
    try:
        root = client.read_run(trace_id)
        actual_trace_id = str(root.trace_id)
        print(f"  Trace ID: {actual_trace_id}")
        print(f"  Name: {root.name}")
        print(f"  Status: {root.status}")
        print(f"  Root total_tokens: {root.total_tokens:,}" if root.total_tokens else "  Root total_tokens: N/A")
        print(f"  Root total_cost: ${float(root.total_cost):.2f}" if root.total_cost else "  Root total_cost: N/A")
    except Exception as e:
        print(f"Error reading run: {e}")
        sys.exit(1)

    # Get all LLM run IDs via trace_id (tokens will be 0 due to API bug)
    print("\nFetching LLM run IDs...")
    all_runs = list(client.list_runs(trace_id=actual_trace_id, run_type="llm"))
    run_ids = [r.id for r in all_runs]
    print(f"  Found {len(run_ids)} LLM runs")

    if not run_ids:
        print("No LLM runs found in trace.")
        return {}

    # Fetch token data via run_ids (workaround for API bug)
    print(f"Fetching token data in batches of {batch_size}...")
    runs_with_tokens = []

    for i in range(0, len(run_ids), batch_size):
        batch = run_ids[i:i+batch_size]
        batch_runs = list(client.list_runs(run_ids=batch))
        runs_with_tokens.extend(batch_runs)
        print(f"  Batch {i//batch_size + 1}/{(len(run_ids) + batch_size - 1)//batch_size}: {len(batch_runs)} runs")

    # Aggregate by model and node
    by_model = defaultdict(lambda: {"input": 0, "output": 0, "cost": 0, "count": 0})
    by_node = defaultdict(lambda: {"input": 0, "output": 0, "cost": 0, "count": 0, "models": set()})
    by_node_model = defaultdict(lambda: {"input": 0, "output": 0, "cost": 0, "count": 0})

    # Filter out wrapper runs that double-count tokens from children
    wrapper_names = {"structured_output", "classify_content"}
    skipped = 0

    for r in runs_with_tokens:
        # Skip wrapper runs - they aggregate child tokens
        if r.name in wrapper_names:
            skipped += 1
            continue

        metadata = r.extra.get("metadata", {}) if r.extra else {}
        model = metadata.get("ls_model_name", "unknown")
        node = metadata.get("langgraph_node", "unknown")

        inp = r.prompt_tokens or 0
        out = r.completion_tokens or 0
        cost = float(r.total_cost) if r.total_cost else 0

        by_model[model]["input"] += inp
        by_model[model]["output"] += out
        by_model[model]["cost"] += cost
        by_model[model]["count"] += 1

        by_node[node]["input"] += inp
        by_node[node]["output"] += out
        by_node[node]["cost"] += cost
        by_node[node]["count"] += 1
        by_node[node]["models"].add(model)

        key = f"{node}|{model}"
        by_node_model[key]["input"] += inp
        by_node_model[key]["output"] += out
        by_node_model[key]["cost"] += cost
        by_node_model[key]["count"] += 1

    return {
        "root": root,
        "by_model": dict(by_model),
        "by_node": dict(by_node),
        "by_node_model": dict(by_node_model),
        "run_count": len(runs_with_tokens),
        "skipped_wrappers": skipped,
    }


def print_report(data: dict):
    """Print formatted token usage report."""
    if not data:
        return

    root = data["root"]
    by_model = data["by_model"]
    by_node = data["by_node"]
    by_node_model = data["by_node_model"]

    # By Model
    print("\n" + "="*100)
    print("TOKEN USAGE BY MODEL")
    print("="*100)
    print(f"{'Model':<40} {'Count':>8} {'Input':>14} {'Output':>14} {'Cost ($)':>12}")
    print("-"*100)

    total_input = total_output = total_cost = 0
    for model, d in sorted(by_model.items(), key=lambda x: x[1]["cost"], reverse=True):
        total_input += d["input"]
        total_output += d["output"]
        total_cost += d["cost"]
        print(f"{model:<40} {d['count']:>8} {d['input']:>14,} {d['output']:>14,} ${d['cost']:>11.2f}")

    print("-"*100)
    total_count = sum(d["count"] for d in by_model.values())
    print(f"{'TOTAL':<40} {total_count:>8} {total_input:>14,} {total_output:>14,} ${total_cost:>11.2f}")

    # By Node
    print("\n" + "="*100)
    print("TOKEN USAGE BY NODE/FUNCTION")
    print("="*100)
    print(f"{'Node':<35} {'Count':>6} {'Input':>12} {'Output':>12} {'Cost ($)':>10} {'Models':<30}")
    print("-"*100)

    for node, d in sorted(by_node.items(), key=lambda x: x[1]["cost"], reverse=True):
        models = ", ".join(sorted(d["models"]))[:30]
        print(f"{node[:35]:<35} {d['count']:>6} {d['input']:>12,} {d['output']:>12,} ${d['cost']:>9.2f} {models}")

    # By Node+Model (detailed)
    print("\n" + "="*100)
    print("DETAILED: NODE + MODEL BREAKDOWN")
    print("="*100)
    print(f"{'Node':<30} {'Model':<30} {'Count':>6} {'Input':>12} {'Output':>12} {'Cost ($)':>10}")
    print("-"*100)

    for key, d in sorted(by_node_model.items(), key=lambda x: x[1]["cost"], reverse=True)[:30]:
        node, model = key.split("|")
        print(f"{node[:30]:<30} {model[:30]:<30} {d['count']:>6} {d['input']:>12,} {d['output']:>12,} ${d['cost']:>9.2f}")

    # Verification
    print("\n" + "="*100)
    print("VERIFICATION")
    print("="*100)
    print(f"LLM runs analyzed: {data['run_count']} (skipped {data.get('skipped_wrappers', 0)} wrapper runs)")
    if root.total_tokens:
        print(f"Root run total_tokens: {root.total_tokens:>14,}")
        print(f"Sum of LLM runs:       {total_input + total_output:>14,}")
        diff = root.total_tokens - (total_input + total_output)
        print(f"Difference:            {diff:>14,} ({diff/root.total_tokens*100:.1f}%)")
    if root.total_cost:
        print(f"\nRoot run total_cost:   ${float(root.total_cost):>13.2f}")
        print(f"Sum of LLM runs:       ${total_cost:>13.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token usage per function for a LangSmith trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("trace_id", help="LangSmith trace ID or run ID")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for fetching runs (default: 100)")

    args = parser.parse_args()

    data = analyze_trace(args.trace_id, args.batch_size)
    print_report(data)


if __name__ == "__main__":
    main()
