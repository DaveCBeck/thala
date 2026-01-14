#!/usr/bin/env python3
"""
Retrieve and analyze a LangSmith run.

Usage:
    python retrieve_langsmith_run.py <run_id>
    python retrieve_langsmith_run.py 019b4588-7b30-7830-b101-ffcd285875e7

Displays:
- Run metadata (status, duration, tokens)
- Supervisor decisions and research questions
- Quality metrics for research workflow analysis
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langsmith import Client


def format_duration(start_time: datetime, end_time: datetime) -> str:
    """Format duration between two timestamps."""
    if not start_time or not end_time:
        return "N/A"
    delta = end_time - start_time
    total_seconds = delta.total_seconds()
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"
    elif total_seconds < 3600:
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def extract_supervisor_decisions(runs: list) -> list[dict]:
    """Extract supervisor decisions from child runs."""
    decisions = []

    for run in runs:
        name = run.name or ""

        # Look for supervisor runs
        if "supervisor" in name.lower():
            decision = {
                "name": name,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "status": run.status,
            }

            # Extract outputs
            if run.outputs:
                outputs = run.outputs
                if "current_status" in outputs:
                    decision["action"] = outputs.get("current_status")
                if "pending_questions" in outputs:
                    questions = outputs.get("pending_questions", [])
                    decision["questions"] = [
                        q.get("question", str(q)) if isinstance(q, dict) else str(q)
                        for q in questions
                    ]
                if "diffusion" in outputs:
                    diffusion = outputs.get("diffusion", {})
                    decision["iteration"] = diffusion.get("iteration")
                    decision["completeness"] = diffusion.get("completeness_score")

            decisions.append(decision)

    return decisions


def extract_researcher_findings(runs: list) -> list[dict]:
    """Extract research findings from researcher runs."""
    findings = []

    for run in runs:
        name = run.name or ""

        # Look for researcher subgraph runs
        if "researcher" in name.lower() and run.outputs:
            outputs = run.outputs
            if "research_findings" in outputs:
                for finding in outputs.get("research_findings", []):
                    findings.append(
                        {
                            "question_id": finding.get("question_id"),
                            "confidence": finding.get("confidence"),
                            "finding_preview": finding.get("finding", "")[:200] + "...",
                            "source_count": len(finding.get("sources", [])),
                            "gaps": finding.get("gaps", []),
                        }
                    )

    return findings


def analyze_run(run_id: str, verbose: bool = False) -> dict:
    """Analyze a LangSmith run and return structured data."""
    client = Client()

    # Get the main run
    run = client.read_run(run_id)

    result = {
        "run_id": run_id,
        "name": run.name,
        "status": run.status,
        "start_time": run.start_time,
        "end_time": run.end_time,
        "duration": format_duration(run.start_time, run.end_time),
        "total_tokens": run.total_tokens,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "error": run.error,
    }

    # Get inputs/outputs
    if run.inputs:
        result["inputs"] = run.inputs
    if run.outputs:
        result["outputs"] = run.outputs

    # Get child runs for detailed analysis
    child_runs = list(
        client.list_runs(
            trace_id=str(run.trace_id),
        )
    )

    result["child_run_count"] = len(child_runs)

    # Extract supervisor decisions
    decisions = extract_supervisor_decisions(child_runs)
    result["supervisor_decisions"] = decisions

    # Extract research findings
    findings = extract_researcher_findings(child_runs)
    result["research_findings"] = findings

    # Compute quality metrics
    result["metrics"] = {
        "total_iterations": len(decisions),
        "total_findings": len(findings),
        "avg_confidence": sum(f.get("confidence", 0) for f in findings) / len(findings)
        if findings
        else 0,
        "total_gaps": sum(len(f.get("gaps", [])) for f in findings),
    }

    if verbose:
        result["all_child_runs"] = [
            {
                "name": r.name,
                "run_type": r.run_type,
                "status": r.status,
                "duration": format_duration(r.start_time, r.end_time),
            }
            for r in child_runs
        ]

    return result


def print_analysis(analysis: dict, verbose: bool = False):
    """Print formatted analysis."""
    print("=" * 80)
    print("LANGSMITH RUN ANALYSIS")
    print("=" * 80)
    print()

    print(f"Run ID: {analysis['run_id']}")
    print(f"Name: {analysis['name']}")
    print(f"Status: {analysis['status']}")
    print(f"Duration: {analysis['duration']}")
    print(f"Total Tokens: {analysis.get('total_tokens', 'N/A')}")
    if analysis.get("error"):
        print(f"Error: {analysis['error']}")
    print()

    # Supervisor decisions
    decisions = analysis.get("supervisor_decisions", [])
    if decisions:
        print("-" * 40)
        print(f"SUPERVISOR DECISIONS ({len(decisions)})")
        print("-" * 40)
        for i, d in enumerate(decisions, 1):
            print(
                f"\n[{i}] {d.get('action', 'unknown')} (iteration {d.get('iteration', '?')})"
            )
            if d.get("completeness") is not None:
                print(f"    Completeness: {d['completeness'] * 100:.0f}%")
            if d.get("questions"):
                print(f"    Questions ({len(d['questions'])}):")
                for q in d["questions"]:
                    # Truncate long questions
                    q_display = q[:100] + "..." if len(q) > 100 else q
                    print(f"      - {q_display}")
        print()

    # Research findings summary
    findings = analysis.get("research_findings", [])
    if findings:
        print("-" * 40)
        print(f"RESEARCH FINDINGS ({len(findings)})")
        print("-" * 40)
        for i, f in enumerate(findings, 1):
            print(
                f"\n[{i}] {f.get('question_id', 'Q?')}: confidence={f.get('confidence', 0):.2f}, sources={f.get('source_count', 0)}"
            )
            if f.get("gaps"):
                print(f"    Gaps: {len(f['gaps'])}")
        print()

    # Metrics
    metrics = analysis.get("metrics", {})
    if metrics:
        print("-" * 40)
        print("QUALITY METRICS")
        print("-" * 40)
        print(f"  Total Iterations: {metrics.get('total_iterations', 0)}")
        print(f"  Total Findings: {metrics.get('total_findings', 0)}")
        print(f"  Avg Confidence: {metrics.get('avg_confidence', 0):.2f}")
        print(f"  Total Gaps: {metrics.get('total_gaps', 0)}")
        print()

    # Verbose: all child runs
    if verbose and analysis.get("all_child_runs"):
        print("-" * 40)
        print(f"ALL CHILD RUNS ({analysis['child_run_count']})")
        print("-" * 40)
        for r in analysis["all_child_runs"]:
            print(f"  {r['name']}: {r['run_type']} - {r['status']} ({r['duration']})")
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve and analyze a LangSmith run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python retrieve_langsmith_run.py 019b4588-7b30-7830-b101-ffcd285875e7
    python retrieve_langsmith_run.py 019b4588-7b30-7830-b101-ffcd285875e7 -v
    python retrieve_langsmith_run.py 019b4588-7b30-7830-b101-ffcd285875e7 --json
        """,
    )
    parser.add_argument("run_id", help="LangSmith run ID to analyze")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show all child runs"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        analysis = analyze_run(args.run_id, verbose=args.verbose)

        if args.json:
            # Convert datetimes for JSON serialization
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")

            print(json.dumps(analysis, indent=2, default=json_serializer))
        else:
            print_analysis(analysis, verbose=args.verbose)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
