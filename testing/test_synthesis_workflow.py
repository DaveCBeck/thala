#!/usr/bin/env python3
"""
Test script for the synthesis workflow.

Runs the complete multi-phase synthesis workflow with LangSmith tracing
enabled for quality analysis.

Usage:
    python test_synthesis_workflow.py "your research topic" [quality] [options]
    python test_synthesis_workflow.py "AI in healthcare" quick
    python test_synthesis_workflow.py "AI in healthcare" standard
    python test_synthesis_workflow.py "AI in healthcare" standard --language es
    python test_synthesis_workflow.py  # Uses default topic and quick quality

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

configure_logging("synthesis")
logger = logging.getLogger(__name__)

# Output directory for results
OUTPUT_DIR = get_output_dir()

VALID_QUALITIES = ["test", "quick", "standard", "comprehensive", "high_quality"]
DEFAULT_QUALITY = "quick"


class SynthesisQualityAnalyzer(BaseQualityAnalyzer):
    """Quality analyzer for synthesis workflow results."""

    output_field = "final_report"
    min_word_count = 2000
    min_source_count = 10

    def _count_sources(self, metrics: QualityMetrics) -> None:
        """Count sources from papers, web research, and books."""
        # Use direct source_count from result if available
        if "source_count" in self.result:
            metrics.source_count = self.result["source_count"]
            return

        # Otherwise calculate from state
        state = load_workflow_state("synthesis", self.result.get("langsmith_run_id"))
        if not state:
            metrics.source_count = 0
            return

        paper_count = len(state.get("paper_corpus", {}))
        web_sources = sum(
            r.get("source_count", 0) for r in state.get("web_research_results", [])
        )
        book_count = len(state.get("selected_books", []))
        metrics.source_count = paper_count + web_sources + book_count

    def _analyze_workflow_specific(self, metrics: QualityMetrics) -> None:
        """Analyze synthesis-specific metrics."""
        state = load_workflow_state("synthesis", self.result.get("langsmith_run_id"))
        if not state:
            return

        # Paper metrics
        paper_corpus = state.get("paper_corpus", {})
        paper_summaries = state.get("paper_summaries", {})
        metrics.workflow_specific["papers_discovered"] = len(paper_corpus)
        metrics.workflow_specific["papers_summarized"] = len(paper_summaries)

        # Web research metrics
        web_results = state.get("web_research_results", [])
        metrics.workflow_specific["web_research_count"] = len(web_results)
        total_web_sources = sum(r.get("source_count", 0) for r in web_results)
        metrics.workflow_specific["web_sources_total"] = total_web_sources

        # Book metrics
        book_results = state.get("book_finding_results", [])
        selected_books = state.get("selected_books", [])
        metrics.workflow_specific["book_searches"] = len(book_results)
        metrics.workflow_specific["books_selected"] = len(selected_books)

        # Structure metrics
        synthesis_structure = state.get("synthesis_structure")
        if synthesis_structure:
            sections = synthesis_structure.get("sections", [])
            metrics.workflow_specific["planned_sections"] = len(sections)

        # Section drafts
        section_drafts = state.get("section_drafts", [])
        metrics.workflow_specific["sections_written"] = len(section_drafts)

        # Zotero integration
        zotero_keys = state.get("zotero_keys", {})
        metrics.workflow_specific["zotero_items"] = len(zotero_keys)

    def _identify_issues(self, metrics: QualityMetrics) -> None:
        """Identify synthesis-specific issues."""
        super()._identify_issues(metrics)

        papers = metrics.workflow_specific.get("papers_discovered", 0)
        if papers < 5:
            metrics.issues.append(f"Low paper count ({papers} papers)")

        web_sources = metrics.workflow_specific.get("web_sources_total", 0)
        if web_sources < 3:
            metrics.issues.append(f"Limited web sources ({web_sources})")

        sections_planned = metrics.workflow_specific.get("planned_sections", 0)
        sections_written = metrics.workflow_specific.get("sections_written", 0)
        if sections_planned > 0 and sections_written < sections_planned:
            metrics.issues.append(
                f"Incomplete sections ({sections_written}/{sections_planned})"
            )

        # Check for errors in result
        errors = self.result.get("errors", [])
        if errors:
            metrics.issues.append(f"Workflow encountered {len(errors)} error(s)")

    def _generate_suggestions(self, metrics: QualityMetrics) -> None:
        """Generate synthesis-specific suggestions."""
        if not metrics.issues:
            metrics.suggestions.append("Synthesis appears comprehensive")
            return

        if "paper count" in str(metrics.issues).lower():
            metrics.suggestions.append(
                "Consider using a higher quality setting for more papers"
            )
        if "web sources" in str(metrics.issues).lower():
            metrics.suggestions.append(
                "Web research may need broader queries or more iterations"
            )
        if "incomplete sections" in str(metrics.issues).lower():
            metrics.suggestions.append("Some sections may have failed - check errors")
        if "short" in str(metrics.issues).lower():
            metrics.suggestions.append(
                "Use higher quality setting for longer synthesis"
            )


def print_result_summary(result: dict, topic: str) -> None:
    """Print a detailed summary of the synthesis result."""
    print_section_header("SYNTHESIS WORKFLOW RESULT")

    # Topic and status
    print(f"\nTopic: {topic}")
    print(f"Status: {result.get('status', 'unknown')}")

    # Timing
    print_timing(result.get("started_at"), result.get("completed_at"))

    # Load state for detailed metrics
    run_id = result.get("langsmith_run_id")
    state = load_workflow_state("synthesis", run_id) if run_id else None

    # Paper Corpus
    paper_corpus = state.get("paper_corpus", {}) if state else {}
    print("\n--- Academic Papers ---")
    print(f"Papers discovered: {len(paper_corpus)}")

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
    print(f"Papers summarized: {len(paper_summaries)}")

    # Web Research
    web_results = state.get("web_research_results", []) if state else []
    print(f"\n--- Web Research ---")
    print(f"Research queries executed: {len(web_results)}")
    total_web_sources = sum(r.get("source_count", 0) for r in web_results)
    print(f"Total web sources found: {total_web_sources}")

    if web_results:
        for i, wr in enumerate(web_results[:3], 1):
            query = wr.get("query", "Unknown")[:50]
            sources = wr.get("source_count", 0)
            print(f"  [{i}] {query}... ({sources} sources)")
        if len(web_results) > 3:
            print(f"  ... and {len(web_results) - 3} more queries")

    # Book Finding
    book_results = state.get("book_finding_results", []) if state else []
    selected_books = state.get("selected_books", []) if state else []
    print(f"\n--- Books ---")
    print(f"Book searches: {len(book_results)}")
    print(f"Books selected: {len(selected_books)}")

    if selected_books:
        for i, book in enumerate(selected_books[:3], 1):
            title = book.get("title", "Unknown")[:50]
            author = book.get("author", "Unknown author")
            print(f"  [{i}] {title}... - {author}")
        if len(selected_books) > 3:
            print(f"  ... and {len(selected_books) - 3} more books")

    # Synthesis Structure
    synthesis_structure = state.get("synthesis_structure") if state else None
    if synthesis_structure:
        sections = synthesis_structure.get("sections", [])
        print(f"\n--- Synthesis Structure ---")
        print(f"Planned sections: {len(sections)}")
        for i, section in enumerate(sections[:5], 1):
            title = section.get("title", "Untitled")[:50]
            print(f"  [{i}] {title}")
        if len(sections) > 5:
            print(f"  ... and {len(sections) - 5} more sections")

    # Section Drafts
    section_drafts = state.get("section_drafts", []) if state else []
    print(f"\n--- Section Drafts ---")
    print(f"Sections written: {len(section_drafts)}")

    # Final Report
    final_report = result.get("final_report", "")
    if final_report:
        print("\n--- Final Report ---")
        word_count = len(final_report.split())
        print(f"Length: {len(final_report)} chars ({word_count} words)")
        print(safe_preview(final_report, 1500))

    # Source Count
    print(f"\n--- Sources ---")
    print(f"Total source count: {result.get('source_count', 0)}")

    # Storage and Tracing
    zotero_keys = state.get("zotero_keys", {}) if state else {}
    langsmith_run_id = result.get("langsmith_run_id")

    print("\n--- Storage & Tracing ---")
    if zotero_keys:
        print(f"Zotero items created: {len(zotero_keys)}")
    if langsmith_run_id:
        print(f"LangSmith Run ID: {langsmith_run_id}")

    # Errors
    print_errors(result.get("errors", []))

    print("\n" + "=" * 80)


async def run_synthesis_test(
    topic: str,
    research_questions: list[str],
    synthesis_brief: str | None = None,
    quality: str = "quick",
    language: str = "en",
) -> dict:
    """Run the synthesis workflow on a topic."""
    from workflows.wrappers.synthesis import synthesis

    logger.info(f"Starting synthesis on: {topic}")
    logger.info(f"Quality: {quality}")
    logger.info(f"Language: {language}")
    logger.info(f"Research questions: {len(research_questions)}")
    logger.info(f"LangSmith tracing: {os.environ.get('LANGSMITH_TRACING', 'false')}")

    result = await synthesis(
        topic=topic,
        research_questions=research_questions,
        synthesis_brief=synthesis_brief,
        quality=quality,
        language=language,
    )

    # Load full state from state store for detailed analysis
    run_id = result.get("langsmith_run_id")
    if run_id:
        full_state = load_workflow_state("synthesis", run_id)
        if full_state:
            logger.info(f"Loaded full state from state store for run {run_id}")
        else:
            logger.warning(
                f"Could not load state for run {run_id} - detailed metrics unavailable"
            )

    return result


def parse_args():
    """Parse command line arguments."""
    parser = create_test_parser(
        description="Run synthesis workflow",
        default_topic="The impact of artificial intelligence on climate change mitigation strategies",
        topic_help="Research topic for synthesis",
        epilog_examples="""
Examples:
  %(prog)s "AI in healthcare"                        # Quick synthesis
  %(prog)s "AI in healthcare" standard               # Standard synthesis
  %(prog)s "quantum computing" comprehensive         # Comprehensive synthesis
  %(prog)s "topic" quick --language es               # Spanish language synthesis
        """,
    )

    add_quality_argument(parser, choices=VALID_QUALITIES, default=DEFAULT_QUALITY)
    add_research_questions_argument(parser)
    add_language_argument(parser)

    parser.add_argument(
        "--brief",
        type=str,
        default=None,
        help="Optional synthesis brief describing desired angle/focus",
    )

    return parser.parse_args()


async def main():
    """Run synthesis workflow test."""
    args = parse_args()

    topic = args.topic
    quality = args.quality
    language = args.language
    synthesis_brief = args.brief

    # Default research questions if not provided
    if args.questions:
        research_questions = args.questions
    else:
        research_questions = [
            f"What are the main developments and trends in {topic}?",
            f"What are the key challenges and opportunities in {topic}?",
            f"What does the research literature say about {topic}?",
        ]

    print_section_header("SYNTHESIS WORKFLOW TEST")
    print(f"\nTopic: {topic}")
    print(f"Quality: {quality}")
    print(f"Language: {language}")
    if synthesis_brief:
        print(f"Brief: {synthesis_brief}")
    print("Research Questions:")
    for q in research_questions:
        print(f"  - {q}")
    print(f"LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'thala-dev')}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        result = await run_synthesis_test(
            topic=topic,
            research_questions=research_questions,
            synthesis_brief=synthesis_brief,
            quality=quality,
            language=language,
        )

        # Print detailed result summary
        print_result_summary(result, topic)

        # Analyze quality
        analyzer = SynthesisQualityAnalyzer(result)
        metrics = analyzer.analyze()
        print_quality_analysis(metrics)

        # Save results
        result_file = save_json_result(result, "synthesis_result")
        logger.info(f"Full result saved to: {result_file}")

        analysis_file = save_json_result(metrics.to_dict(), "synthesis_analysis")
        logger.info(f"Analysis saved to: {analysis_file}")

        # Save final report as markdown
        if result.get("final_report"):
            report_file = save_markdown_report(
                result["final_report"],
                "synthesis",
                title=f"Synthesis: {topic}",
                metadata={"quality": quality, "language": language},
            )
            logger.info(f"Report saved to: {report_file}")

        return result, metrics.to_dict()

    except Exception as e:
        logger.error(f"Synthesis test failed: {e}", exc_info=True)
        return {"error": str(e)}, {"error": str(e)}


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Wait for LangSmith to flush all trace data before exiting
        # Without this, the process may exit before end_time is sent,
        # causing runs to appear as "interrupted" in LangSmith
        wait_for_all_tracers()
