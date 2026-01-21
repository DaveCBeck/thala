#!/usr/bin/env python3
"""
Test script for the academic literature review workflow.

Runs a comprehensive literature review workflow on a given topic with LangSmith
tracing enabled for quality analysis.

Usage:
    python test_academic_lit_review.py "your research topic" [quality] [options]
    python test_academic_lit_review.py "transformer architectures" quick
    python test_academic_lit_review.py "transformer architectures" standard
    python test_academic_lit_review.py "transformer architectures" standard --language es
    python test_academic_lit_review.py  # Uses default topic and quick quality

Valid quality levels: test, quick, standard, comprehensive, high_quality (default: quick)
Valid languages: en, es, zh, ja, de, fr, pt, ko, ru, ar, it, nl, pl, tr, vi, th, id, hi, bn, sv, no, da, fi, cs, el, he, uk, ro, hu (default: en)

Environment:
    Set THALA_MODE=dev in .env to enable LangSmith tracing
"""

import asyncio
import os
from datetime import datetime

# Enable dev mode for LangSmith tracing before any imports
os.environ["THALA_MODE"] = "dev"

import logging

from langchain_core.tracers.langchain import wait_for_all_tracers

from testing.utils import (
    BaseQualityAnalyzer,
    QualityMetrics,
    add_date_range_arguments,
    add_language_argument,
    add_quality_argument,
    add_research_questions_argument,
    configure_logging,
    create_test_parser,
    get_output_dir,
    print_errors,
    print_quality_analysis,
    print_section_header,
    print_timing,
    safe_preview,
    save_json_result,
    save_markdown_report,
)
from workflows.shared.workflow_state_store import load_workflow_state

configure_logging("academic_lit_review")
logger = logging.getLogger(__name__)

# Output directory for results
OUTPUT_DIR = get_output_dir()

VALID_QUALITIES = ["test", "quick", "standard", "comprehensive", "high_quality"]
DEFAULT_QUALITY = "quick"


class AcademicLitReviewQualityAnalyzer(BaseQualityAnalyzer):
    """Quality analyzer for academic literature review results."""

    output_field = "final_review"
    min_word_count = 1000
    min_source_count = 5

    def _count_sources(self, metrics: QualityMetrics) -> None:
        """Count sources from paper corpus and references."""
        state = load_workflow_state(
            "academic_lit_review", self.result["langsmith_run_id"]
        )
        paper_corpus = state.get("paper_corpus", {}) if state else {}
        references = state.get("references", []) if state else []
        metrics.source_count = max(len(paper_corpus), len(references))

    def _analyze_workflow_specific(self, metrics: QualityMetrics) -> None:
        """Analyze academic lit review specific metrics."""
        state = load_workflow_state(
            "academic_lit_review", self.result["langsmith_run_id"]
        )

        # Paper metrics
        paper_corpus = state.get("paper_corpus", {}) if state else {}
        paper_summaries = state.get("paper_summaries", {}) if state else {}

        metrics.workflow_specific["papers_discovered"] = len(paper_corpus)
        metrics.workflow_specific["papers_processed"] = len(paper_summaries)

        if paper_corpus:
            metrics.workflow_specific["processing_rate"] = len(paper_summaries) / len(
                paper_corpus
            )

        # Clustering metrics
        clusters = state.get("clusters", []) if state else []
        metrics.workflow_specific["cluster_count"] = len(clusters)

        if clusters:
            papers_clustered = sum(len(c.get("paper_dois", [])) for c in clusters)
            metrics.workflow_specific["papers_clustered"] = papers_clustered
            metrics.workflow_specific["gaps_identified"] = sum(
                len(c.get("gaps", [])) for c in clusters
            )
            metrics.workflow_specific["conflicts_identified"] = sum(
                len(c.get("conflicts", [])) for c in clusters
            )

        # Diffusion metrics
        diffusion = state.get("diffusion", {}) if state else {}
        if diffusion:
            metrics.workflow_specific["diffusion_stages"] = diffusion.get(
                "current_stage", 0
            )
            metrics.workflow_specific["saturation_reached"] = diffusion.get(
                "is_saturated", False
            )

        # References
        references = state.get("references", []) if state else []
        metrics.workflow_specific["reference_count"] = len(references)

    def _identify_issues(self, metrics: QualityMetrics) -> None:
        """Identify academic-specific issues."""
        super()._identify_issues(metrics)

        papers = metrics.workflow_specific.get("papers_discovered", 0)
        if papers < 10:
            metrics.issues.append(f"Low paper count ({papers} papers)")

        processing_rate = metrics.workflow_specific.get("processing_rate", 1)
        if processing_rate < 0.5:
            metrics.issues.append(f"Low processing rate ({processing_rate:.0%})")

        clusters = metrics.workflow_specific.get("cluster_count", 0)
        if clusters < 3:
            metrics.issues.append(f"Few clusters identified ({clusters})")

        refs = metrics.workflow_specific.get("reference_count", 0)
        if refs < 5:
            metrics.issues.append(f"Few references ({refs})")

    def _generate_suggestions(self, metrics: QualityMetrics) -> None:
        """Generate academic-specific suggestions."""
        if not metrics.issues:
            metrics.suggestions.append("Literature review appears comprehensive")
            return

        if "paper count" in str(metrics.issues).lower():
            metrics.suggestions.append(
                "Consider broadening search terms or using higher quality setting"
            )
        if "clusters" in str(metrics.issues).lower():
            metrics.suggestions.append("More papers needed for meaningful clustering")
        if "short" in str(metrics.issues).lower():
            metrics.suggestions.append("Use higher quality setting for longer reviews")


def print_result_summary(result: dict, topic: str) -> None:
    """Print a detailed summary of the literature review result."""
    print_section_header("ACADEMIC LITERATURE REVIEW RESULT")

    # Topic
    print(f"\nTopic: {topic}")

    # Timing
    print_timing(result.get("started_at"), result.get("completed_at"))

    # Load state for removed fields
    run_id = result.get("langsmith_run_id")
    state = load_workflow_state("academic_lit_review", run_id) if run_id else None

    # Paper Corpus
    paper_corpus = state.get("paper_corpus", {}) if state else {}
    print("\n--- Paper Corpus ---")
    print(f"Total papers discovered: {len(paper_corpus)}")

    if paper_corpus:
        sample_papers = list(paper_corpus.items())[:5]
        print("Sample papers:")
        for doi, paper in sample_papers:
            title = paper.get("title", "Unknown")[:60]
            year = paper.get("year", "?")
            citations = paper.get("cited_by_count", 0)
            print(f"  - [{year}] {title}... ({citations} citations)")
        if len(paper_corpus) > 5:
            print(f"  ... and {len(paper_corpus) - 5} more papers")

    # Paper Summaries
    paper_summaries = state.get("paper_summaries", {}) if state else {}
    print("\n--- Paper Summaries ---")
    print(f"Papers processed: {len(paper_summaries)}")

    # Diffusion State
    diffusion = state.get("diffusion", {}) if state else {}
    if diffusion:
        print("\n--- Diffusion Algorithm ---")
        print(
            f"Stages completed: {diffusion.get('current_stage', 0)}/{diffusion.get('max_stages', 'N/A')}"
        )
        print(f"Is saturated: {diffusion.get('is_saturated', False)}")
        print(f"Total discovered: {diffusion.get('total_papers_discovered', 0)}")
        print(f"Total relevant: {diffusion.get('total_papers_relevant', 0)}")
        print(f"Total rejected: {diffusion.get('total_papers_rejected', 0)}")

    # Clusters
    clusters = state.get("clusters", []) if state else []
    print(f"\n--- Thematic Clusters ({len(clusters)}) ---")
    for i, cluster in enumerate(clusters[:10], 1):
        label = cluster.get("label", "Unnamed")
        paper_count = len(cluster.get("paper_dois", []))
        description = cluster.get("description", "")[:80]
        print(f"\n  [{i}] {label} ({paper_count} papers)")
        print(f"      {description}...")
        sub_themes = cluster.get("sub_themes", [])
        if sub_themes:
            print(f"      Sub-themes: {', '.join(sub_themes[:3])}")
        gaps = cluster.get("gaps", [])
        if gaps:
            print(f"      Gaps: {gaps[0][:60]}...")
    if len(clusters) > 10:
        print(f"\n  ... and {len(clusters) - 10} more clusters")

    # Final Review
    final_review = result.get("final_review", "")
    if final_review:
        print("\n--- Final Review ---")
        word_count = len(final_review.split())
        print(f"Length: {len(final_review)} chars ({word_count} words)")
        print(safe_preview(final_review, 1500))

    # References
    references = state.get("references", []) if state else []
    if references:
        print(f"\n--- References ({len(references)}) ---")
        for i, ref in enumerate(references[:10], 1):
            citation = ref.get("citation_text", "Unknown")[:80]
            print(f"  [{i}] {citation}...")
        if len(references) > 10:
            print(f"  ... and {len(references) - 10} more references")

    # PRISMA Documentation
    prisma = state.get("prisma_documentation", "") if state else ""
    if prisma:
        print("\n--- PRISMA Documentation ---")
        print(safe_preview(prisma, 500))

    # Storage and Tracing
    zotero_keys = state.get("zotero_keys", {}) if state else {}
    es_ids = state.get("elasticsearch_ids", {}) if state else {}
    langsmith_run_id = result.get("langsmith_run_id")

    print("\n--- Storage & Tracing ---")
    if zotero_keys:
        print(f"Zotero items created: {len(zotero_keys)}")
    if es_ids:
        print(f"Elasticsearch records: {len(es_ids)}")
    if langsmith_run_id:
        print(f"LangSmith Run ID: {langsmith_run_id}")

    # Errors
    print_errors(result.get("errors", []))

    print("\n" + "=" * 80)


async def run_literature_review(
    topic: str,
    research_questions: list[str],
    quality: str = "quick",
    date_range: tuple[int, int] | None = None,
    language: str = "en",
) -> dict:
    """Run the academic literature review workflow on a topic."""
    from workflows.research.academic_lit_review import academic_lit_review

    logger.info(f"Starting literature review on: {topic}")
    logger.info(f"Quality: {quality}")
    logger.info(f"Language: {language}")
    logger.info(f"Research questions: {len(research_questions)}")
    logger.info(f"LangSmith tracing: {os.environ.get('LANGSMITH_TRACING', 'false')}")

    result = await academic_lit_review(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        date_range=date_range,
        language=language,
    )

    # Load full state from state store for detailed analysis
    run_id = result.get("langsmith_run_id")
    if run_id:
        full_state = load_workflow_state("academic_lit_review", run_id)
        if full_state:
            # Merge full state into result for analysis/display
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
        description="Run academic literature review workflow",
        default_topic="The impact of large language models on software engineering practices",
        topic_help="Research topic for literature review",
        epilog_examples="""
Examples:
  %(prog)s "transformer architectures"              # Quick review
  %(prog)s "transformer architectures" standard     # Standard review
  %(prog)s "AI in drug discovery" comprehensive     # Comprehensive review
  %(prog)s "topic" quick --language es              # Spanish language review
        """,
    )

    add_quality_argument(parser, choices=VALID_QUALITIES, default=DEFAULT_QUALITY)
    add_research_questions_argument(parser)
    add_date_range_arguments(parser)
    add_language_argument(parser)

    return parser.parse_args()


async def main():
    """Run academic literature review test."""
    args = parse_args()

    topic = args.topic
    quality = args.quality
    language = args.language

    # Default research questions if not provided
    if args.questions:
        research_questions = args.questions
    else:
        research_questions = [
            f"What are the main research themes in {topic}?",
            f"What methodological approaches are used to study {topic}?",
            f"What are the key findings and debates in {topic}?",
        ]

    # Date range
    date_range = None
    if args.from_year or args.to_year:
        from_year = args.from_year or 2000
        to_year = args.to_year or 2025
        date_range = (from_year, to_year)

    print_section_header("ACADEMIC LITERATURE REVIEW TEST")
    print(f"\nTopic: {topic}")
    print(f"Quality: {quality}")
    print(f"Language: {language}")
    print("Research Questions:")
    for q in research_questions:
        print(f"  - {q}")
    if date_range:
        print(f"Date Range: {date_range[0]}-{date_range[1]}")
    print(f"LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'thala-dev')}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        result = await run_literature_review(
            topic=topic,
            research_questions=research_questions,
            quality=quality,
            date_range=date_range,
            language=language,
        )

        # Print detailed result summary
        print_result_summary(result, topic)

        # Analyze quality
        analyzer = AcademicLitReviewQualityAnalyzer(result)
        metrics = analyzer.analyze()
        print_quality_analysis(metrics)

        # Save results
        result_file = save_json_result(result, "lit_review_result")
        logger.info(f"Full result saved to: {result_file}")

        analysis_file = save_json_result(metrics.to_dict(), "lit_review_analysis")
        logger.info(f"Analysis saved to: {analysis_file}")

        # Save final review as markdown
        if result.get("final_review"):
            report_file = save_markdown_report(
                result["final_review"],
                "lit_review",
                title=f"Literature Review: {topic}",
                metadata={"quality": quality, "language": language},
            )
            logger.info(f"Review (v1) saved to: {report_file}")

        # Save supervised review (v2) if present
        if result.get("final_review_v2"):
            report_v2_file = save_markdown_report(
                result["final_review_v2"],
                "lit_review_v2",
                title=f"Literature Review (Supervised): {topic}",
                metadata={"quality": quality, "language": language},
            )
            logger.info(f"Review (v2) saved to: {report_v2_file}")

        # Save PRISMA documentation
        if result.get("prisma_documentation"):
            prisma_file = save_markdown_report(
                result["prisma_documentation"],
                "lit_review_prisma",
            )
            logger.info(f"PRISMA docs saved to: {prisma_file}")

        return result, metrics.to_dict()

    except Exception as e:
        logger.error(f"Literature review failed: {e}", exc_info=True)
        return {"error": str(e)}, {"error": str(e)}


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Wait for LangSmith to flush all trace data before exiting
        wait_for_all_tracers()
