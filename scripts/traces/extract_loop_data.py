#!/usr/bin/env python3
"""
Extract detailed data for each supervision loop for analysis.
Creates separate files for each loop execution.
"""

import json
import sys
from pathlib import Path


def extract_loop_data(trace_file: str, output_dir: str):
    """Extract detailed loop data into separate files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(trace_file) as f:
        trace = json.load(f)

    # Build parent-child relationships
    parent_to_children = {}
    id_to_run = {}
    for run in trace["child_runs"]:
        id_to_run[run["id"]] = run
        parent_id = run.get("parent_run_id")
        if parent_id:
            if parent_id not in parent_to_children:
                parent_to_children[parent_id] = []
            parent_to_children[parent_id].append(run)

    def get_descendants(run_id, depth=0, max_depth=10):
        """Get all descendants of a run up to max_depth."""
        if depth > max_depth:
            return []
        descendants = []
        for child in parent_to_children.get(run_id, []):
            descendants.append(
                {
                    "depth": depth,
                    "id": child["id"],
                    "name": child["name"],
                    "run_type": child["run_type"],
                    "status": child["status"],
                    "inputs": child.get("inputs"),
                    "outputs": child.get("outputs"),
                    "error": child.get("error"),
                }
            )
            descendants.extend(get_descendants(child["id"], depth + 1, max_depth))
        return descendants

    # Loop names to extract
    loop_names = ["loop3", "loop4", "loop4_5", "loop5", "holistic_review"]

    for run in trace["child_runs"]:
        if run["name"] in loop_names:
            loop_data = {
                "name": run["name"],
                "id": run["id"],
                "start_time": run.get("start_time"),
                "end_time": run.get("end_time"),
                "status": run["status"],
                "inputs": run.get("inputs"),
                "outputs": run.get("outputs"),
                "descendants": get_descendants(run["id"]),
            }

            # Create filename with timestamp
            timestamp = (
                run.get("start_time", "unknown").replace(":", "-").replace(".", "-")
            )
            filename = f"{run['name']}_{timestamp}.json"
            filepath = output_path / filename

            with open(filepath, "w") as f:
                json.dump(loop_data, f, indent=2)

            print(f"Wrote {filepath} ({len(loop_data['descendants'])} descendants)")

    # Also extract paper corpus for reference checking
    corpus_data = trace["main_run"]["inputs"].get("paper_corpus", {})
    with open(output_path / "paper_corpus.json", "w") as f:
        json.dump(corpus_data, f, indent=2)
    print(f"Wrote paper_corpus.json ({len(corpus_data)} papers)")

    # Extract the current_review (input) and any final outputs
    with open(output_path / "input_review.md", "w") as f:
        f.write(trace["main_run"]["inputs"].get("current_review", ""))
    print("Wrote input_review.md")

    # Extract final output if available
    final_output = trace["main_run"].get("outputs", {})
    with open(output_path / "final_output.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print("Wrote final_output.json")


if __name__ == "__main__":
    trace_file = (
        sys.argv[1] if len(sys.argv) > 1 else "testing/traces/supervision_trace.json"
    )
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "testing/traces/loops"
    extract_loop_data(trace_file, output_dir)
