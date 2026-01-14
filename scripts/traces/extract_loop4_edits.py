#!/usr/bin/env python3
"""Extract all edits made by Loop 4 from LangSmith trace."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def find_all_runs(obj: Any, runs_list: list[dict]) -> None:
    """Recursively find all runs in trace structure."""
    if isinstance(obj, dict):
        if "parent_run_id" in obj and "id" in obj:
            runs_list.append(obj)
        for value in obj.values():
            find_all_runs(value, runs_list)
    elif isinstance(obj, list):
        for item in obj:
            find_all_runs(item, runs_list)


def extract_original_from_inputs(run: dict, section_id: str | None) -> str | None:
    """Extract original section text from ChatAnthropic input messages.

    The original section content is in the input message with format:
    ## Your Section to Edit: {section_id}
    {section_content}
    """
    inputs = run.get("inputs", {})
    messages = inputs.get("messages", [])

    for msg in messages:
        # Handle both dict and list message formats
        content = None
        if isinstance(msg, dict):
            content = msg.get("content", "")
            # Also check kwargs.content for nested structure
            if not content and "kwargs" in msg:
                content = msg.get("kwargs", {}).get("content", "")
        elif isinstance(msg, list):
            # LangChain serialization: list of message components
            # e.g., [system_msg, human_msg] where each has kwargs.content
            for item in msg:
                if isinstance(item, dict):
                    # Check kwargs.content (LangChain format)
                    item_content = item.get("kwargs", {}).get("content", "")
                    if item_content and isinstance(item_content, str):
                        if "## Your Section to Edit:" in item_content:
                            content = item_content
                            break
                    # Also check direct content
                    if not content:
                        item_content = item.get("content", "")
                        if item_content and isinstance(item_content, str):
                            if "## Your Section to Edit:" in item_content:
                                content = item_content
                                break

        if not content or not isinstance(content, str):
            continue

        # Look for the section marker
        marker = "## Your Section to Edit:"
        if marker not in content:
            continue

        # Find the section content after the marker
        marker_pos = content.find(marker)
        if marker_pos == -1:
            continue

        # Extract from after the marker line to the next ## marker
        section_start = content.find("\n", marker_pos)
        if section_start == -1:
            continue

        section_start += 1  # Skip the newline

        # Find the end (next ## marker or end of content)
        next_section = content.find("\n## ", section_start)
        if next_section == -1:
            original_text = content[section_start:].strip()
        else:
            original_text = content[section_start:next_section].strip()

        return original_text if original_text else None

    return None


def extract_edits_from_run(run: dict) -> list[dict]:
    """Extract edits from a ChatAnthropic run's outputs."""
    edits = []
    outputs = run.get("outputs")

    if not outputs or not isinstance(outputs, dict):
        return edits

    generations = outputs.get("generations", [[]])

    if not generations or not generations[0]:
        return edits

    for generation in generations[0]:
        message = generation.get("message", {})
        content = message.get("kwargs", {}).get("content", [])

        if isinstance(content, str):
            continue

        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                if item.get("name") == "submit_result":
                    input_data = item.get("input", {})
                    section_id = input_data.get("section_id")

                    # Extract original text from input messages
                    original_text = extract_original_from_inputs(run, section_id)

                    edit = {
                        "section_id": section_id,
                        "section_title": extract_title_from_content(
                            input_data.get("edited_content", "")
                        ),
                        "original": original_text,
                        "edited": input_data.get("edited_content"),
                        "confidence": input_data.get("confidence"),
                        "notes": input_data.get("notes"),
                        "new_paper_todos": input_data.get("new_paper_todos", []),
                        "edit_type": input_data.get("edit_type"),
                        "status": "accepted",  # All submitted edits are accepted
                        "rejection_reason": None,
                        "citations_added": extract_citations(
                            input_data.get("edited_content", "")
                        ),
                        "run_id": run.get("id"),
                    }
                    edits.append(edit)

    return edits


def extract_title_from_content(content: str) -> str | None:
    """Extract section title from markdown content."""
    if not content:
        return None

    lines = content.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            # Remove markdown heading markers
            return line.lstrip("#").strip()

    return None


def extract_citations(content: str) -> list[str]:
    """Extract citation keys from markdown content."""
    if not content:
        return []

    import re

    # Match [@XXXXXX] pattern
    pattern = r"\[@([A-Z0-9]+)\]"
    matches = re.findall(pattern, content)
    return list(set(matches))


def extract_loop4_edits(trace_path: Path) -> dict:
    """Extract all edits from Loop 4 iterations."""
    with open(trace_path) as f:
        data = json.load(f)

    # Find all runs
    all_runs = []
    find_all_runs(data, all_runs)

    # Find loop4 runs
    loop4_runs = [r for r in all_runs if r.get("name") == "loop4"]
    loop4_runs.sort(key=lambda x: x.get("start_time", ""))

    if len(loop4_runs) != 2:
        print(f"Warning: Expected 2 loop4 runs, found {len(loop4_runs)}")

    # Extract edits from each iteration
    results = {
        "trace_id": data.get("run_id"),
        "extraction_date": datetime.now().isoformat(),
        "iterations": [],
    }

    for idx, loop4_run in enumerate(loop4_runs, 1):
        loop4_id = loop4_run.get("id")
        print(f"Processing iteration {idx}: {loop4_id}")

        # Find LangGraph child
        langgraph_runs = [
            r
            for r in all_runs
            if r.get("parent_run_id") == loop4_id and r.get("name") == "LangGraph"
        ]

        if not langgraph_runs:
            print(f"  No LangGraph child found for {loop4_id}")
            continue

        langgraph_id = langgraph_runs[0].get("id")

        # Find parallel_edit_sections child
        parallel_edit_runs = [
            r
            for r in all_runs
            if r.get("parent_run_id") == langgraph_id
            and r.get("name") == "parallel_edit_sections"
        ]

        if not parallel_edit_runs:
            print(f"  No parallel_edit_sections found for LangGraph {langgraph_id}")
            continue

        parallel_edit_id = parallel_edit_runs[0].get("id")

        # Find all ChatAnthropic children of parallel_edit_sections
        chat_runs = [
            r
            for r in all_runs
            if r.get("parent_run_id") == parallel_edit_id
            and r.get("name") == "ChatAnthropic"
        ]

        print(f"  Found {len(chat_runs)} ChatAnthropic runs")

        # Extract edits from each ChatAnthropic run
        iteration_edits = []
        for chat_run in chat_runs:
            edits = extract_edits_from_run(chat_run)
            iteration_edits.extend(edits)

        print(f"  Extracted {len(iteration_edits)} edits")

        results["iterations"].append(
            {
                "iteration": idx,
                "run_id": loop4_id,
                "start_time": loop4_run.get("start_time"),
                "end_time": loop4_run.get("end_time"),
                "total_edits": len(iteration_edits),
                "edits": iteration_edits,
            }
        )

    return results


def main():
    trace_path = Path(
        "/home/dave/thala/testing/traces/019ba18e-7124-7923-9dca-4565ead80738.json"
    )
    output_path = Path("/home/dave/thala/testing/traces/loop4_edits.json")

    print(f"Extracting Loop 4 edits from {trace_path}")

    results = extract_loop4_edits(trace_path)

    print("\nExtraction complete:")
    print(f"  Total iterations: {len(results['iterations'])}")
    for iteration in results["iterations"]:
        print(f"  Iteration {iteration['iteration']}: {iteration['total_edits']} edits")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
