#!/usr/bin/env python3
"""
Test script for the research workflow.

Runs a comprehensive research workflow on a given topic with LangSmith tracing
enabled for quality analysis.

Usage:
    python test_research_workflow.py "your research topic here" [quality] [options]
    python test_research_workflow.py "AI agents" quick
    python test_research_workflow.py "AI agents" comprehensive
    python test_research_workflow.py  # Uses default topic and standard quality

    # Language options:
    python test_research_workflow.py "topic" standard --language es

Valid quality tiers: quick, standard, comprehensive (default: standard)

Environment:
    Set THALA_MODE=dev in .env to enable LangSmith tracing
"""

import asyncio
import os
from datetime import datetime

# Enable dev mode for LangSmith tracing before any imports
os.environ["THALA_MODE"] = "dev"

import logging

from testing.utils import (
    configure_logging,
    get_output_dir,
    save_json_result,
    save_markdown_report,
    print_section_header,
    safe_preview,
    print_timing,
    print_key_value,
    print_list_preview,
    print_errors,
    print_quality_analysis,
    BaseQualityAnalyzer,
    QualityMetrics,
    create_test_parser,
    add_quality_argument,
    add_language_argument,
)
from workflows.shared.workflow_state_store import load_workflow_state

configure_logging("research_workflow")
logger = logging.getLogger(__name__)

# Output directory for results
OUTPUT_DIR = get_output_dir()

VALID_DEPTHS = ["quick", "standard", "comprehensive"]
DEFAULT_DEPTH = "standard"


class ResearchQualityAnalyzer(BaseQualityAnalyzer):
    """Quality analyzer for research workflow results."""

    output_field = "final_report"
    min_word_count = 500
    max_word_count = 5000
    min_source_count = 5

    def _count_sources(self, metrics: QualityMetrics) -> None:
        """Count sources from findings and citations."""
        state = load_workflow_state("web_research", self.result["langsmith_run_id"])

        findings = state.get("research_findings", []) if state else []
        total_sources = sum(len(f.get("sources", [])) for f in findings)

        citations = state.get("citations", []) if state else []
        metrics.source_count = max(total_sources, len(citations))

    def _analyze_workflow_specific(self, metrics: QualityMetrics) -> None:
        """Analyze research-specific metrics."""
        state = load_workflow_state("web_research", self.result["langsmith_run_id"])

        findings = state.get("research_findings", []) if state else []
        metrics.workflow_specific["total_findings"] = len(findings)

        # Confidence scores
        confidence_scores = [f.get("confidence", 0) for f in findings]
        if confidence_scores:
            metrics.workflow_specific["avg_confidence"] = sum(confidence_scores) / len(
                confidence_scores
            )

        # Gaps
        all_gaps = []
        for finding in findings:
            all_gaps.extend(finding.get("gaps", []))
        metrics.workflow_specific["unique_gaps"] = len(set(all_gaps))

        # Diffusion metrics
        diffusion = state.get("diffusion", {}) if state else {}
        if diffusion:
            metrics.workflow_specific["completeness_score"] = diffusion.get(
                "completeness_score", 0
            )
            metrics.workflow_specific["iterations_used"] = diffusion.get("iteration", 0)
            metrics.workflow_specific["max_iterations"] = diffusion.get(
                "max_iterations", 0
            )
            metrics.workflow_specific["areas_explored"] = len(
                diffusion.get("areas_explored", [])
            )
            metrics.workflow_specific["areas_remaining"] = len(
                diffusion.get("areas_to_explore", [])
            )

        # Translation
        translated_report = state.get("translated_report") if state else None
        if translated_report:
            metrics.workflow_specific["translation_generated"] = True
            metrics.workflow_specific["translation_length"] = len(translated_report)

        # Citations
        citations = state.get("citations", []) if state else []
        metrics.workflow_specific["citation_count"] = len(citations)

    def _identify_issues(self, metrics: QualityMetrics) -> None:
        """Identify research-specific issues."""
        super()._identify_issues(metrics)

        avg_conf = metrics.workflow_specific.get("avg_confidence", 0)
        if avg_conf and avg_conf < 0.6:
            metrics.issues.append(f"Low average confidence ({avg_conf:.0%})")

        completeness = metrics.workflow_specific.get("completeness_score", 0)
        if completeness and completeness < 0.7:
            metrics.issues.append(f"Low completeness score ({completeness:.0%})")

        areas_remaining = metrics.workflow_specific.get("areas_remaining", 0)
        if areas_remaining:
            state = load_workflow_state("web_research", self.result["langsmith_run_id"])
            diffusion = state.get("diffusion", {}) if state else {}
            areas = diffusion.get("areas_to_explore", [])[:3]
            metrics.issues.append(f"Unexplored areas: {', '.join(areas)}")

    def _generate_suggestions(self, metrics: QualityMetrics) -> None:
        """Generate research-specific suggestions."""
        if not metrics.issues:
            metrics.suggestions.append(
                "Research appears comprehensive - no major issues detected"
            )
            return

        if "source count" in str(metrics.issues).lower():
            metrics.suggestions.append(
                "Consider increasing max_sources or adding more search sources"
            )
        if "confidence" in str(metrics.issues).lower():
            metrics.suggestions.append(
                "Improve source quality filtering or add domain-specific sources"
            )
        if "completeness" in str(metrics.issues).lower():
            metrics.suggestions.append(
                "Consider increasing max_iterations for more thorough research"
            )
        if "Unexplored" in str(metrics.issues):
            metrics.suggestions.append(
                "Diffusion algorithm may need tuning for better coverage"
            )


def print_result_summary(result: dict, topic: str) -> None:
    """Print a detailed summary of the research workflow result."""
    print_section_header("RESEARCH WORKFLOW RESULT")

    # Load state once for all field accesses
    run_id = result.get("langsmith_run_id")
    state = load_workflow_state("web_research", run_id) if run_id else None

    # Topic and Status
    print(f"\nTopic: {topic}")
    status = result.get("current_status", "unknown")
    print(f"Status: {status}")

    # Timing
    print_timing(result.get("started_at"), result.get("completed_at"))

    # Research Brief
    brief = state.get("research_brief") if state else None
    if brief:
        print("\n--- Research Brief ---")
        print_key_value("Topic", brief.get("topic", "N/A"))
        print_key_value("Scope", brief.get("scope", "N/A"))
        print_list_preview(
            brief.get("objectives", []),
            "Objectives",
            format_item=str,
        )
        print_list_preview(
            brief.get("key_questions", []),
            "Key Questions",
            format_item=str,
        )

    # Memory Context
    memory_context = state.get("memory_context") if state else None
    if memory_context:
        print("\n--- Memory Context ---")
        print(safe_preview(memory_context, 500))

    # Research Plan
    research_plan = result.get("research_plan")
    if research_plan:
        print("\n--- Research Plan ---")
        print(safe_preview(research_plan, 800))

    # Diffusion State
    diffusion = state.get("diffusion") if state else None
    if diffusion:
        print("\n--- Diffusion Algorithm ---")
        print(
            f"Iterations: {diffusion.get('iteration', 0)}/{diffusion.get('max_iterations', 'N/A')}"
        )
        print(f"Completeness Score: {diffusion.get('completeness_score', 0):.1%}")
        print_list_preview(
            diffusion.get("areas_explored", []),
            "Areas Explored",
            max_items=10,
        )

    # Research Findings
    findings = state.get("research_findings", []) if state else []
    if findings:
        print(f"\n--- Research Findings ({len(findings)}) ---")
        for i, finding in enumerate(findings[:5], 1):
            conf = finding.get("confidence", 0)
            sources = finding.get("sources", [])
            print(f"\n  [{i}] Confidence: {conf:.0%}, Sources: {len(sources)}")
            finding_text = safe_preview(finding.get("finding", ""), 300, suffix="...")
            print(f"      {finding_text}")
            gaps = finding.get("gaps", [])
            if gaps:
                print(f"      Gaps: {', '.join(gaps[:3])}")
        if len(findings) > 5:
            print(f"\n  ... and {len(findings) - 5} more findings")

    # Draft Report
    draft = state.get("draft_report") if state else None
    if draft:
        print(f"\n--- Draft Report (v{draft.get('version', 0)}) ---")
        print(f"Length: {len(draft.get('content', ''))} chars")
        gaps = draft.get("gaps_remaining", [])
        if gaps:
            print(f"Remaining Gaps: {len(gaps)}")
            for gap in gaps[:3]:
                print(f"  - {gap}")

    # Final Report
    final_report = result.get("final_report")
    if final_report:
        print("\n--- Final Report ---")
        print(f"Length: {len(final_report)} chars ({len(final_report.split())} words)")
        print(safe_preview(final_report, 1000))

    # Citations
    citations = state.get("citations", []) if state else []
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
    store_id = state.get("store_record_id") if state else None
    zotero_key = result.get("zotero_key")
    langsmith_run_id = result.get("langsmith_run_id")
    if store_id or zotero_key or langsmith_run_id:
        print("\n--- Storage ---")
        if store_id:
            print(f"Store Record ID: {store_id}")
        if zotero_key:
            print(f"Zotero Key: {zotero_key}")
        if langsmith_run_id:
            print(f"LangSmith Run ID: {langsmith_run_id}")

    # Language Results
    primary_lang = result.get("primary_language")
    translated_report = state.get("translated_report") if state else None

    if primary_lang and primary_lang != "en":
        print("\n--- Language Settings ---")
        print(f"Primary Language: {primary_lang}")

    if translated_report:
        print("\n--- Translated Report ---")
        print(
            f"Length: {len(translated_report)} chars ({len(translated_report.split())} words)"
        )
        print(safe_preview(translated_report, 800))

    # Errors
    print_errors(result.get("errors", []))

    print("\n" + "=" * 80)


async def run_research(
    topic: str,
    quality: str = "standard",
    language: str = None,
) -> dict:
    """Run the research workflow on a topic.

    Args:
        topic: Research question or topic
        quality: Research quality tier (quick, standard, comprehensive)
        language: Single language mode (ISO 639-1 code, e.g., "es", "zh")
    """
    from workflows.research.web_research import deep_research

    logger.info(f"Starting research on: {topic}")
    logger.info(f"Quality: {quality}")
    if language:
        logger.info(f"Language: {language}")
    logger.info(f"LangSmith tracing: {os.environ.get('LANGSMITH_TRACING', 'false')}")

    result = await deep_research(
        query=topic,
        quality=quality,
        language=language,
    )

    # Load full state from state store for detailed analysis
    run_id = result.get("langsmith_run_id")
    if run_id:
        full_state = load_workflow_state("web_research", run_id)
        if full_state:
            result = {**full_state, **result}
            logger.info(f"Loaded full state from state store for run {run_id}")
        else:
            logger.warning(
                f"Could not load state for run {run_id} - detailed metrics unavailable"
            )

    return result


def parse_args():
    """Parse command line arguments."""
    parser = create_test_parser(
        description="Run research workflow with optional language support",
        default_topic="What are the current best practices for building reliable AI agents, particularly around tool use, error handling, and human-in-the-loop patterns?",
        topic_help="Research topic or question",
        epilog_examples="""
Examples:
  %(prog)s "AI agents"                          # Standard English research
  %(prog)s "AI agents" quick                    # Quick research
  %(prog)s "AI agents" --language es            # Research in Spanish
        """,
    )

    add_quality_argument(parser, choices=VALID_DEPTHS, default=DEFAULT_DEPTH)
    add_language_argument(parser)

    return parser.parse_args()


async def main():
    """Run research workflow test."""
    args = parse_args()

    topic = args.topic
    quality = args.quality

    print_section_header("RESEARCH WORKFLOW TEST")
    print(f"\nTopic: {topic}")
    print(f"Quality: {quality}")
    if args.language:
        print(f"Language: {args.language}")
    print(f"LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'thala-dev')}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        result = await run_research(
            topic=topic,
            quality=quality,
            language=args.language,
        )

        # Print detailed result summary
        print_result_summary(result, topic)

        # Analyze quality
        analyzer = ResearchQualityAnalyzer(result)
        metrics = analyzer.analyze()
        print_quality_analysis(metrics)

        # Save results
        result_file = save_json_result(result, "research_result")
        logger.info(f"Full result saved to: {result_file}")

        analysis_file = save_json_result(metrics.to_dict(), "research_analysis")
        logger.info(f"Analysis saved to: {analysis_file}")

        # Save final report as markdown
        if result.get("final_report"):
            report_file = save_markdown_report(
                result["final_report"],
                "research_report",
                title=topic,
            )
            logger.info(f"Report saved to: {report_file}")

        return result, metrics.to_dict()

    except Exception as e:
        logger.error(f"Research failed: {e}", exc_info=True)
        return {"error": str(e)}, {"error": str(e)}

    finally:
        # Clean up resources
        from workflows.research.web_research import cleanup_research_resources

        try:
            logger.info("Cleaning up research resources...")
            await cleanup_research_resources()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())
