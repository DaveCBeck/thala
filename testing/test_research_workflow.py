#!/usr/bin/env python3
"""
Test script for the research workflow.

Runs a comprehensive research workflow on a given topic with LangSmith tracing
enabled for quality analysis.

Usage:
    python test_research_workflow.py "your research topic here" [depth]
    python test_research_workflow.py "AI agents" quick
    python test_research_workflow.py "AI agents" comprehensive
    python test_research_workflow.py  # Uses default topic and standard depth

Valid depths: quick, standard, comprehensive (default: standard)

Environment:
    Set THALA_MODE=dev in .env to enable LangSmith tracing
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Enable dev mode for LangSmith tracing before any imports
os.environ["THALA_MODE"] = "dev"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Output directory for results
OUTPUT_DIR = Path(__file__).parent / "test_data"

VALID_DEPTHS = ["quick", "standard", "comprehensive"]
DEFAULT_DEPTH = "standard"


def print_result_summary(result: dict, topic: str) -> None:
    """Print a detailed summary of the research workflow result."""
    print("\n" + "=" * 80)
    print(f"RESEARCH WORKFLOW RESULT")
    print("=" * 80)

    # Topic and Status
    print(f"\nTopic: {topic}")
    status = result.get("current_status", "unknown")
    print(f"Status: {status}")

    # Timing
    started = result.get("started_at")
    completed = result.get("completed_at")
    if started and completed:
        try:
            if isinstance(started, str):
                started = datetime.fromisoformat(started.replace("Z", "+00:00"))
            if isinstance(completed, str):
                completed = datetime.fromisoformat(completed.replace("Z", "+00:00"))
            if hasattr(started, 'replace') and started.tzinfo is not None:
                started = started.replace(tzinfo=None)
            if hasattr(completed, 'replace') and completed.tzinfo is not None:
                completed = completed.replace(tzinfo=None)
            duration = (completed - started).total_seconds()
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            print(f"Duration: {minutes}m {seconds}s ({duration:.1f}s total)")
        except Exception as e:
            print(f"Duration: (error calculating: {e})")

    # Research Brief
    brief = result.get("research_brief")
    if brief:
        print(f"\n--- Research Brief ---")
        print(f"Topic: {brief.get('topic', 'N/A')}")
        print(f"Scope: {brief.get('scope', 'N/A')}")
        objectives = brief.get("objectives", [])
        if objectives:
            print(f"Objectives ({len(objectives)}):")
            for obj in objectives[:5]:
                print(f"  - {obj}")
        key_questions = brief.get("key_questions", [])
        if key_questions:
            print(f"Key Questions ({len(key_questions)}):")
            for q in key_questions[:5]:
                print(f"  - {q}")

    # Memory Context
    memory_context = result.get("memory_context")
    if memory_context:
        print(f"\n--- Memory Context ---")
        preview = memory_context[:500] + "..." if len(memory_context) > 500 else memory_context
        print(preview)

    # Research Plan
    research_plan = result.get("research_plan")
    if research_plan:
        print(f"\n--- Research Plan ---")
        preview = research_plan[:800] + "..." if len(research_plan) > 800 else research_plan
        print(preview)

    # Diffusion State
    diffusion = result.get("diffusion")
    if diffusion:
        print(f"\n--- Diffusion Algorithm ---")
        print(f"Iterations: {diffusion.get('iteration', 0)}/{diffusion.get('max_iterations', 'N/A')}")
        print(f"Completeness Score: {diffusion.get('completeness_score', 0):.1%}")
        areas_explored = diffusion.get("areas_explored", [])
        if areas_explored:
            print(f"Areas Explored ({len(areas_explored)}):")
            for area in areas_explored[:10]:
                print(f"  - {area}")

    # Research Findings
    findings = result.get("research_findings", [])
    if findings:
        print(f"\n--- Research Findings ({len(findings)}) ---")
        for i, finding in enumerate(findings[:5], 1):
            conf = finding.get("confidence", 0)
            sources = finding.get("sources", [])
            print(f"\n  [{i}] Confidence: {conf:.0%}, Sources: {len(sources)}")
            finding_text = finding.get("finding", "")[:300]
            if len(finding.get("finding", "")) > 300:
                finding_text += "..."
            print(f"      {finding_text}")
            gaps = finding.get("gaps", [])
            if gaps:
                print(f"      Gaps: {', '.join(gaps[:3])}")
        if len(findings) > 5:
            print(f"\n  ... and {len(findings) - 5} more findings")

    # Draft Report (if available)
    draft = result.get("draft_report")
    if draft:
        print(f"\n--- Draft Report (v{draft.get('version', 0)}) ---")
        content = draft.get("content", "")
        print(f"Length: {len(content)} chars")
        gaps = draft.get("gaps_remaining", [])
        if gaps:
            print(f"Remaining Gaps: {len(gaps)}")
            for gap in gaps[:3]:
                print(f"  - {gap}")

    # Final Report
    final_report = result.get("final_report")
    if final_report:
        print(f"\n--- Final Report ---")
        print(f"Length: {len(final_report)} chars ({len(final_report.split())} words)")
        # Show first 1000 chars
        preview = final_report[:1000]
        if len(final_report) > 1000:
            preview += "\n\n... [truncated] ..."
        print(preview)

    # Citations
    citations = result.get("citations", [])
    if citations:
        print(f"\n--- Citations ({len(citations)}) ---")
        for i, cite in enumerate(citations[:10], 1):
            title = cite.get("title", "Untitled")[:60]
            url = cite.get("url", "N/A")[:50]
            print(f"  [{i}] {title}")
            print(f"      {url}")
        if len(citations) > 10:
            print(f"  ... and {len(citations) - 10} more citations")

    # Storage and Tracing
    store_id = result.get("store_record_id")
    zotero_key = result.get("zotero_key")
    langsmith_run_id = result.get("langsmith_run_id")
    if store_id or zotero_key or langsmith_run_id:
        print(f"\n--- Storage ---")
        if store_id:
            print(f"Store Record ID: {store_id}")
        if zotero_key:
            print(f"Zotero Key: {zotero_key}")
        if langsmith_run_id:
            print(f"LangSmith Run ID: {langsmith_run_id}")

    # Errors
    errors = result.get("errors", [])
    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for err in errors:
            node = err.get("node", "unknown")
            error = err.get("error", "unknown")
            print(f"  [{node}]: {error}")

    print("\n" + "=" * 80)


def analyze_quality(result: dict) -> dict:
    """Analyze research quality and return metrics for improvement."""
    analysis = {
        "metrics": {},
        "issues": [],
        "suggestions": [],
    }

    # Basic completion metrics
    status = result.get("current_status", "")
    analysis["metrics"]["completed"] = status == "completed"

    # Report quality
    final_report = result.get("final_report", "")
    if final_report:
        word_count = len(final_report.split())
        analysis["metrics"]["report_word_count"] = word_count
        analysis["metrics"]["report_char_count"] = len(final_report)

        if word_count < 500:
            analysis["issues"].append("Report is short (< 500 words)")
        elif word_count > 5000:
            analysis["issues"].append("Report may be too long (> 5000 words)")
    else:
        analysis["issues"].append("No final report generated")

    # Source coverage
    findings = result.get("research_findings", [])
    analysis["metrics"]["total_findings"] = len(findings)

    total_sources = 0
    confidence_scores = []
    all_gaps = []
    for finding in findings:
        sources = finding.get("sources", [])
        total_sources += len(sources)
        conf = finding.get("confidence", 0)
        confidence_scores.append(conf)
        gaps = finding.get("gaps", [])
        all_gaps.extend(gaps)

    analysis["metrics"]["total_sources"] = total_sources
    analysis["metrics"]["avg_confidence"] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    analysis["metrics"]["unique_gaps"] = len(set(all_gaps))

    if total_sources < 5:
        analysis["issues"].append("Low source count (< 5 sources)")

    if analysis["metrics"]["avg_confidence"] < 0.6:
        analysis["issues"].append(f"Low average confidence ({analysis['metrics']['avg_confidence']:.0%})")

    # Diffusion metrics
    diffusion = result.get("diffusion", {})
    if diffusion:
        completeness = diffusion.get("completeness_score", 0)
        iterations = diffusion.get("iteration", 0)
        max_iters = diffusion.get("max_iterations", 0)

        analysis["metrics"]["completeness_score"] = completeness
        analysis["metrics"]["iterations_used"] = iterations
        analysis["metrics"]["max_iterations"] = max_iters
        analysis["metrics"]["iteration_efficiency"] = iterations / max_iters if max_iters else 0

        if completeness < 0.7:
            analysis["issues"].append(f"Low completeness score ({completeness:.0%})")

        areas_explored = diffusion.get("areas_explored", [])
        areas_to_explore = diffusion.get("areas_to_explore", [])
        analysis["metrics"]["areas_explored"] = len(areas_explored)
        analysis["metrics"]["areas_remaining"] = len(areas_to_explore)

        if areas_to_explore:
            analysis["issues"].append(f"Unexplored areas: {', '.join(areas_to_explore[:3])}")

    # Citations
    citations = result.get("citations", [])
    analysis["metrics"]["citation_count"] = len(citations)

    if len(citations) < 3:
        analysis["issues"].append("Insufficient citations (< 3)")

    # Errors
    errors = result.get("errors", [])
    analysis["metrics"]["error_count"] = len(errors)
    if errors:
        analysis["issues"].append(f"{len(errors)} errors encountered during research")

    # Generate suggestions
    if not analysis["issues"]:
        analysis["suggestions"].append("Research appears comprehensive - no major issues detected")
    else:
        if "Low source count" in str(analysis["issues"]):
            analysis["suggestions"].append("Consider increasing max_sources or adding more search sources")
        if "Low average confidence" in str(analysis["issues"]):
            analysis["suggestions"].append("Improve source quality filtering or add domain-specific sources")
        if "Low completeness" in str(analysis["issues"]):
            analysis["suggestions"].append("Consider increasing max_iterations for more thorough research")
        if "Unexplored areas" in str(analysis["issues"]):
            analysis["suggestions"].append("Diffusion algorithm may need tuning for better coverage")

    return analysis


def print_quality_analysis(analysis: dict) -> None:
    """Print quality analysis summary."""
    print("\n" + "=" * 80)
    print("QUALITY ANALYSIS")
    print("=" * 80)

    # Metrics
    print("\n--- Metrics ---")
    metrics = analysis.get("metrics", {})
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Issues
    issues = analysis.get("issues", [])
    if issues:
        print(f"\n--- Issues Found ({len(issues)}) ---")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n--- No Issues Found ---")

    # Suggestions
    suggestions = analysis.get("suggestions", [])
    if suggestions:
        print(f"\n--- Suggestions ---")
        for suggestion in suggestions:
            print(f"  - {suggestion}")

    print("\n" + "=" * 80)


async def run_research(topic: str, depth: str = "standard") -> dict:
    """Run the research workflow on a topic."""
    from workflows.research import deep_research

    logger.info(f"Starting research on: {topic}")
    logger.info(f"Depth: {depth}")
    logger.info(f"LangSmith tracing: {os.environ.get('LANGSMITH_TRACING', 'false')}")

    result = await deep_research(
        query=topic,
        depth=depth,
    )

    return result


async def main():
    """Run research workflow test."""
    # Default topic if none provided
    default_topic = "What are the current best practices for building reliable AI agents, particularly around tool use, error handling, and human-in-the-loop patterns?"

    topic = sys.argv[1] if len(sys.argv) > 1 else default_topic
    depth = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DEPTH

    # Validate depth
    if depth not in VALID_DEPTHS:
        print(f"Error: Invalid depth '{depth}'. Must be one of: {', '.join(VALID_DEPTHS)}")
        print(f"Usage: python test_research_workflow.py \"topic\" [depth]")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("RESEARCH WORKFLOW TEST")
    print(f"{'=' * 80}")
    print(f"\nTopic: {topic}")
    print(f"Depth: {depth}")
    print(f"LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'thala-dev')}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        result = await run_research(topic, depth)

        # Print detailed result summary
        print_result_summary(result, topic)

        # Analyze quality
        analysis = analyze_quality(result)
        print_quality_analysis(analysis)

        # Save results
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save full result
        result_file = OUTPUT_DIR / f"research_result_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Full result saved to: {result_file}")

        # Save analysis
        analysis_file = OUTPUT_DIR / f"research_analysis_{timestamp}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to: {analysis_file}")

        # Save final report as markdown
        if result.get("final_report"):
            report_file = OUTPUT_DIR / f"research_report_{timestamp}.md"
            with open(report_file, "w") as f:
                f.write(f"# {topic}\n\n")
                f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write(result["final_report"])
            logger.info(f"Report saved to: {report_file}")

        return result, analysis

    except Exception as e:
        logger.error(f"Research failed: {e}", exc_info=True)
        return {"error": str(e)}, {"error": str(e)}

    finally:
        # Clean up resources
        from workflows.research import cleanup_research_resources
        try:
            logger.info("Cleaning up research resources...")
            await cleanup_research_resources()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())
