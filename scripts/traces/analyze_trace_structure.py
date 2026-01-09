#!/usr/bin/env python3
"""
Analyze the structure of a supervision trace to identify loop executions and their hierarchy.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def analyze_trace(trace_file: str):
    """Analyze trace structure and extract loop information."""
    with open(trace_file) as f:
        trace = json.load(f)

    print(f"Trace: {trace['meta']['name']}")
    print(f"Status: {trace['meta']['status']}")
    print(f"Total tokens: {trace['meta']['total_tokens']:,}")
    print(f"Child runs: {trace['meta']['child_run_count']}")
    print()

    # Group runs by name
    runs_by_name = defaultdict(list)
    for run in trace['child_runs']:
        runs_by_name[run['name']].append(run)

    # Identify loop-related runs
    loop_names = ['loop3', 'loop4', 'loop4_5', 'loop5', 'holistic_review',
                  'parallel_edit_sections', 'analyze_structure', 'number_paragraphs',
                  'execute_manifest', 'validate_result', 'reassemble_document',
                  'split_sections']

    print("=" * 80)
    print("SUPERVISION LOOP EXECUTIONS")
    print("=" * 80)

    for name in loop_names:
        if name in runs_by_name:
            runs = runs_by_name[name]
            print(f"\n{name}: {len(runs)} execution(s)")
            for i, run in enumerate(runs):
                print(f"  [{i+1}] Status: {run['status']}, Tokens: {run.get('total_tokens', 'N/A')}")
                # Show brief input/output summary
                if run.get('inputs'):
                    input_keys = list(run['inputs'].keys())[:5]
                    print(f"      Inputs: {input_keys}")
                if run.get('outputs'):
                    output_keys = list(run['outputs'].keys())[:5]
                    print(f"      Outputs: {output_keys}")

    # Look for LLM calls with specific patterns
    print("\n" + "=" * 80)
    print("LLM CALLS BREAKDOWN")
    print("=" * 80)

    llm_runs = runs_by_name.get('ChatAnthropic', [])
    print(f"\nTotal ChatAnthropic calls: {len(llm_runs)}")

    # Analyze parent relationships to group by loop
    loop_llm_counts = defaultdict(int)
    for run in llm_runs:
        parent_id = run.get('parent_run_id')
        # Find the parent run name
        for other_run in trace['child_runs']:
            if other_run['id'] == parent_id:
                loop_llm_counts[other_run['name']] += 1
                break

    print("\nLLM calls by parent node:")
    for name, count in sorted(loop_llm_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {name}: {count}")

    # Extract loop execution timeline
    print("\n" + "=" * 80)
    print("LOOP EXECUTION TIMELINE")
    print("=" * 80)

    loop_timeline = []
    for name in ['loop3', 'loop4', 'loop4_5', 'loop5', 'holistic_review']:
        if name in runs_by_name:
            for run in runs_by_name[name]:
                loop_timeline.append({
                    'name': name,
                    'start': run.get('start_time'),
                    'end': run.get('end_time'),
                    'status': run.get('status')
                })

    loop_timeline.sort(key=lambda x: x['start'] or '')
    for item in loop_timeline:
        print(f"  {item['start']} - {item['name']} ({item['status']})")

    # Look for store access patterns
    print("\n" + "=" * 80)
    print("STORE ACCESS PATTERNS")
    print("=" * 80)

    store_refs = []
    for run in trace['child_runs']:
        inputs_str = json.dumps(run.get('inputs', {}))
        outputs_str = json.dumps(run.get('outputs', {}))

        if 'store' in inputs_str.lower() or 'store' in outputs_str.lower():
            store_refs.append(run['name'])
        if 'corpus' in inputs_str.lower() or 'corpus' in outputs_str.lower():
            store_refs.append(f"{run['name']} (corpus)")
        if 'paper_' in inputs_str.lower() or 'paper_' in outputs_str.lower():
            store_refs.append(f"{run['name']} (paper ref)")

    store_counts = defaultdict(int)
    for ref in store_refs:
        store_counts[ref] += 1

    print("\nNodes with store/corpus references:")
    for name, count in sorted(store_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {name}: {count}")

    return trace


if __name__ == "__main__":
    trace_file = sys.argv[1] if len(sys.argv) > 1 else "testing/traces/supervision_trace.json"
    analyze_trace(trace_file)
