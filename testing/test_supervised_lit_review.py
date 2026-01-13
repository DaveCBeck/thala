#!/usr/bin/env python3
"""
Test script for the supervised academic literature review workflow.

Runs a comprehensive literature review workflow with multi-loop supervision
on a given topic with LangSmith tracing enabled for quality analysis.

The supervised workflow runs the academic_lit_review first, then applies
configurable supervision loops to enhance quality:
- Loop 1: Citation coverage & missing sources
- Loop 2: Structural editing & logical flow
- Loop 3: Fact-checking & claim verification
- Loop 4: Gap analysis & completeness

Usage:
    python test_supervised_lit_review.py "your research topic" [quality] [options]
    python test_supervised_lit_review.py "transformer architectures" quick
    python test_supervised_lit_review.py "transformer architectures" standard --loops all
    python test_supervised_lit_review.py "transformer architectures" standard --loops two
    python test_supervised_lit_review.py "transformer architectures" standard --language es
    python test_supervised_lit_review.py  # Uses default topic and quick quality

Valid quality levels: test, quick, standard, comprehensive, high_quality (default: quick)
Valid supervision loops: none, one, two, three, four, all (default: all)
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

from testing.utils import (
    configure_logging,
    get_output_dir,
    save_json_result,
    save_markdown_report,
    print_section_header,
    safe_preview,
    print_timing,
    print_errors,
    print_quality_analysis,
    create_test_parser,
    add_quality_argument,
    add_language_argument,
    add_date_range_arguments,
    add_research_questions_argument,
)
from workflows.shared.workflow_state_store import load_workflow_state

configure_logging("supervised_lit_review")
logger = logging.getLogger(__name__)

# Output directory for results
OUTPUT_DIR = get_output_dir()

VALID_QUALITIES = ["test", "quick", "standard", "comprehensive", "high_quality"]
DEFAULT_QUALITY = "quick"

VALID_LOOPS = ["none", "one", "two", "three", "four", "all"]
DEFAULT_LOOPS = "all"


def print_result_summary(result: dict, topic: str) -> None:
    """Print a detailed summary of the supervised literature review result."""
    print_section_header("SUPERVISED LITERATURE REVIEW RESULT")

    # Topic
    print(f"\nTopic: {topic}")

    # Status
    status = result.get("status", "unknown")
    print(f"Status: {status}")

    # Timing
    print_timing(result.get("started_at"), result.get("completed_at"))

    # Supervision Status
    supervision = result.get("supervision", {})
    if supervision:
        print(f"\n--- Supervision ---")
        loops_run = supervision.get("loops_run", [])
        print(f"Loops run: {', '.join(loops_run) if loops_run else 'None'}")
        completion_reason = supervision.get("completion_reason", "N/A")
        print(f"Completion reason: {completion_reason}")
        new_papers = supervision.get("new_papers_added", 0)
        print(f"New papers added: {new_papers}")
        loop_progress = supervision.get("loop_progress")
        if loop_progress:
            print("Loop progress:")
            for loop_name, progress in loop_progress.items():
                print(f"  - {loop_name}: {progress}")

    # Human Review Items
    human_review = result.get("human_review_items", [])
    if human_review:
        print(f"\n--- Human Review Items ({len(human_review)}) ---")
        for i, item in enumerate(human_review[:5], 1):
            # Handle both dict items and string items
            if isinstance(item, dict):
                item_type = item.get("type", "unknown")
                description = item.get("description", "No description")[:80]
                print(f"  [{i}] ({item_type}) {description}...")
            else:
                # Item is a string
                print(f"  [{i}] {str(item)[:80]}...")
        if len(human_review) > 5:
            print(f"  ... and {len(human_review) - 5} more items")

    # Paper Corpus
    paper_corpus = result.get("paper_corpus", {})
    print(f"\n--- Paper Corpus ---")
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
    paper_summaries = result.get("paper_summaries", {})
    print(f"\n--- Paper Summaries ---")
    print(f"Papers processed: {len(paper_summaries)}")

    # Diffusion State
    diffusion = result.get("diffusion", {})
    if diffusion:
        print(f"\n--- Diffusion Algorithm ---")
        print(f"Stages completed: {diffusion.get('current_stage', 0)}/{diffusion.get('max_stages', 'N/A')}")
        print(f"Is saturated: {diffusion.get('is_saturated', False)}")
        print(f"Total discovered: {diffusion.get('total_papers_discovered', 0)}")
        print(f"Total relevant: {diffusion.get('total_papers_relevant', 0)}")
        print(f"Total rejected: {diffusion.get('total_papers_rejected', 0)}")

    # Clusters
    clusters = result.get("clusters", [])
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

    # Original Review (v1)
    final_review = result.get("final_review", "")
    if final_review:
        print(f"\n--- Original Review (v1) ---")
        word_count = len(final_review.split())
        print(f"Length: {len(final_review)} chars ({word_count} words)")

    # Intermediate Loop Reviews
    loop_names = {
        1: "Theoretical Depth",
        2: "Literature Expansion",
        3: "Structural Editing",
        4: "Section Editing",
    }
    for loop_num in [1, 2, 3, 4]:
        key = f"review_loop{loop_num}"
        review_text = result.get(key, "")
        if review_text:
            word_count = len(review_text.split())
            print(f"\n--- Review after Loop {loop_num} ({loop_names[loop_num]}) ---")
            print(f"Length: {len(review_text)} chars ({word_count} words)")

    # Supervised Review (v2)
    final_review_v2 = result.get("final_review_v2", "")
    if final_review_v2:
        print(f"\n--- Supervised Review (v2) ---")
        word_count = len(final_review_v2.split())
        print(f"Length: {len(final_review_v2)} chars ({word_count} words)")
        print(safe_preview(final_review_v2, 1500))
    elif final_review:
        print(f"\n--- Final Review (no supervision applied) ---")
        word_count = len(final_review.split())
        print(f"Length: {len(final_review)} chars ({word_count} words)")
        print(safe_preview(final_review, 1500))

    # Standardized final_report field
    final_report = result.get("final_report", "")
    if final_report and final_report != final_review_v2 and final_report != final_review:
        print(f"\n--- Final Report (standardized) ---")
        word_count = len(final_report.split())
        print(f"Length: {len(final_report)} chars ({word_count} words)")

    # References
    references = result.get("references", [])
    if references:
        print(f"\n--- References ({len(references)}) ---")
        for i, ref in enumerate(references[:10], 1):
            citation = ref.get("citation_text", "Unknown")[:80]
            print(f"  [{i}] {citation}...")
        if len(references) > 10:
            print(f"  ... and {len(references) - 10} more references")

    # PRISMA Documentation
    prisma = result.get("prisma_documentation", "")
    if prisma:
        print(f"\n--- PRISMA Documentation ---")
        print(safe_preview(prisma, 500))

    # Storage and Tracing
    zotero_keys = result.get("zotero_keys", {})
    es_ids = result.get("elasticsearch_ids", {})
    langsmith_run_id = result.get("langsmith_run_id")

    print(f"\n--- Storage & Tracing ---")
    if zotero_keys:
        print(f"Zotero items created: {len(zotero_keys)}")
    if es_ids:
        print(f"Elasticsearch records: {len(es_ids)}")
    if langsmith_run_id:
        print(f"LangSmith Run ID: {langsmith_run_id}")

    # Errors
    print_errors(result.get("errors", []))

    print("\n" + "=" * 80)


def analyze_quality(result: dict) -> dict:
    """Analyze supervised literature review quality and return metrics."""
    analysis = {
        "metrics": {},
        "issues": [],
        "suggestions": [],
    }

    # Status
    status = result.get("status", "unknown")
    analysis["metrics"]["status"] = status
    analysis["metrics"]["completed"] = status in ["success", "partial"]

    # Review metrics
    final_review = result.get("final_review", "")
    final_review_v2 = result.get("final_review_v2", "")

    if final_review:
        word_count = len(final_review.split())
        analysis["metrics"]["v1_word_count"] = word_count
        analysis["metrics"]["v1_char_count"] = len(final_review)

    if final_review_v2:
        word_count_v2 = len(final_review_v2.split())
        analysis["metrics"]["v2_word_count"] = word_count_v2
        analysis["metrics"]["v2_char_count"] = len(final_review_v2)
        analysis["metrics"]["supervision_applied"] = True

        # Calculate improvement
        if final_review:
            word_improvement = word_count_v2 - len(final_review.split())
            analysis["metrics"]["word_count_change"] = word_improvement
    else:
        analysis["metrics"]["supervision_applied"] = False
        if final_review:
            analysis["issues"].append("Supervision did not produce v2 review")

    if not final_review and not final_review_v2:
        analysis["issues"].append("No review generated")

    # Supervision metrics
    supervision = result.get("supervision", {})
    if supervision:
        loops_run = supervision.get("loops_run", [])
        analysis["metrics"]["loops_completed"] = len(loops_run)
        analysis["metrics"]["loops_list"] = loops_run
        analysis["metrics"]["new_papers_added"] = supervision.get("new_papers_added", 0)
        analysis["metrics"]["completion_reason"] = supervision.get("completion_reason", "unknown")

        if supervision.get("error"):
            analysis["issues"].append(f"Supervision error: {supervision['error']}")
    else:
        analysis["metrics"]["loops_completed"] = 0

    # Human review items
    human_review = result.get("human_review_items", [])
    analysis["metrics"]["human_review_items"] = len(human_review)
    if len(human_review) > 10:
        analysis["issues"].append(f"Many items flagged for human review ({len(human_review)})")

    # Paper coverage
    paper_corpus = result.get("paper_corpus", {})
    paper_summaries = result.get("paper_summaries", {})
    analysis["metrics"]["papers_discovered"] = len(paper_corpus)
    analysis["metrics"]["papers_processed"] = len(paper_summaries)

    processing_rate = len(paper_summaries) / len(paper_corpus) if paper_corpus else 0
    analysis["metrics"]["processing_rate"] = processing_rate

    if len(paper_corpus) < 10:
        analysis["issues"].append(f"Low paper count ({len(paper_corpus)} papers)")

    if processing_rate < 0.5:
        analysis["issues"].append(f"Low processing rate ({processing_rate:.0%})")

    # Clustering metrics
    clusters = result.get("clusters", [])
    analysis["metrics"]["cluster_count"] = len(clusters)

    if clusters:
        papers_clustered = sum(len(c.get("paper_dois", [])) for c in clusters)
        analysis["metrics"]["papers_clustered"] = papers_clustered

        gaps_found = sum(len(c.get("gaps", [])) for c in clusters)
        conflicts_found = sum(len(c.get("conflicts", [])) for c in clusters)
        analysis["metrics"]["gaps_identified"] = gaps_found
        analysis["metrics"]["conflicts_identified"] = conflicts_found

    if len(clusters) < 3:
        analysis["issues"].append(f"Few clusters identified ({len(clusters)})")

    # Diffusion metrics
    diffusion = result.get("diffusion", {})
    if diffusion:
        analysis["metrics"]["diffusion_stages"] = diffusion.get("current_stage", 0)
        analysis["metrics"]["saturation_reached"] = diffusion.get("is_saturated", False)

    # References
    references = result.get("references", [])
    analysis["metrics"]["reference_count"] = len(references)

    if len(references) < 5:
        analysis["issues"].append(f"Few references ({len(references)})")

    # Errors
    errors = result.get("errors", [])
    analysis["metrics"]["error_count"] = len(errors)
    if errors:
        analysis["issues"].append(f"{len(errors)} errors encountered")

    # Generate suggestions
    if not analysis["issues"]:
        analysis["suggestions"].append("Supervised literature review appears comprehensive")
    else:
        if "Low paper count" in str(analysis["issues"]):
            analysis["suggestions"].append("Consider broadening search terms or using higher quality setting")
        if "Few clusters" in str(analysis["issues"]):
            analysis["suggestions"].append("More papers needed for meaningful clustering")
        if not analysis["metrics"].get("supervision_applied"):
            analysis["suggestions"].append("Check supervision configuration; no supervision was applied")
        if "Many items flagged" in str(analysis["issues"]):
            analysis["suggestions"].append("Review flagged items to improve review quality")

    return analysis


async def run_supervised_literature_review(
    topic: str,
    research_questions: list[str],
    quality: str = "quick",
    date_range: tuple[int, int] | None = None,
    language: str = "en",
    supervision_loops: str = "all",
) -> dict:
    """Run the supervised academic literature review workflow on a topic."""
    from workflows.supervised_lit_review import supervised_lit_review

    logger.info(f"Starting supervised literature review on: {topic}")
    logger.info(f"Quality: {quality}")
    logger.info(f"Language: {language}")
    logger.info(f"Supervision loops: {supervision_loops}")
    logger.info(f"Research questions: {len(research_questions)}")
    logger.info(f"LangSmith tracing: {os.environ.get('LANGSMITH_TRACING', 'false')}")

    result = await supervised_lit_review(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        date_range=date_range,
        language=language,
        supervision_loops=supervision_loops,
    )

    # Load full state from state store for detailed analysis
    run_id = result.get("langsmith_run_id")
    if run_id:
        full_state = load_workflow_state("supervised_lit_review", run_id)
        if full_state:
            result = {**full_state, **result}
            logger.info(f"Loaded full state from state store for run {run_id}")
        else:
            logger.warning(f"Could not load state for run {run_id} - detailed metrics unavailable")

    return result


def parse_args():
    """Parse command line arguments."""
    parser = create_test_parser(
        description="Run supervised academic literature review workflow",
        default_topic="The impact of large language models on software engineering practices",
        topic_help="Research topic for literature review",
        epilog_examples="""
Examples:
  %(prog)s "transformer architectures"                    # Quick review with all supervision
  %(prog)s "transformer architectures" standard           # Standard review with all loops
  %(prog)s "AI in drug discovery" comprehensive           # Comprehensive review
  %(prog)s "topic" standard --loops two                   # Run only first two supervision loops
  %(prog)s "topic" standard --loops none                  # Skip supervision (same as academic_lit_review)

Supervision loops:
  none   - No supervision (equivalent to academic_lit_review)
  one    - Citation coverage loop only
  two    - Citation + structural editing
  three  - Citation + structural + fact-checking
  four   - Citation + structural + fact-checking + gap analysis
  all    - All supervision loops (default)
        """
    )

    add_quality_argument(parser, choices=VALID_QUALITIES, default=DEFAULT_QUALITY)
    add_research_questions_argument(parser)
    add_date_range_arguments(parser)
    add_language_argument(parser)

    parser.add_argument(
        "--loops",
        type=str,
        choices=VALID_LOOPS,
        default=DEFAULT_LOOPS,
        help=f"Supervision loops to run (default: {DEFAULT_LOOPS})"
    )

    return parser.parse_args()


async def main():
    """Run supervised academic literature review test."""
    args = parse_args()

    topic = args.topic
    quality = args.quality
    language = args.language
    supervision_loops = args.loops

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

    print_section_header("SUPERVISED ACADEMIC LITERATURE REVIEW TEST")
    print(f"\nTopic: {topic}")
    print(f"Quality: {quality}")
    print(f"Language: {language}")
    print(f"Supervision loops: {supervision_loops}")
    print(f"Research Questions:")
    for q in research_questions:
        print(f"  - {q}")
    if date_range:
        print(f"Date Range: {date_range[0]}-{date_range[1]}")
    print(f"LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'thala-dev')}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        result = await run_supervised_literature_review(
            topic=topic,
            research_questions=research_questions,
            quality=quality,
            date_range=date_range,
            language=language,
            supervision_loops=supervision_loops,
        )

        # Print detailed result summary
        print_result_summary(result, topic)

        # Analyze quality
        analysis = analyze_quality(result)
        print_quality_analysis(analysis)

        # Save results
        result_file = save_json_result(result, "supervised_lit_review_result")
        logger.info(f"Full result saved to: {result_file}")

        analysis_file = save_json_result(analysis, "supervised_lit_review_analysis")
        logger.info(f"Analysis saved to: {analysis_file}")

        # Save original review (v1, before supervision) as markdown
        if result.get("final_review"):
            report_file = save_markdown_report(
                result["final_review"],
                "supervised_lit_review_v1",
                title=f"Literature Review (Original): {topic}",
                metadata={
                    "quality": quality,
                    "language": language,
                    "supervision": supervision_loops,
                },
            )
            logger.info(f"Original review (v1) saved to: {report_file}")

        # Save intermediate loop reviews as markdown
        loop_descriptions = {
            1: "theoretical_depth",
            2: "literature_expansion",
            3: "structural_editing",
            4: "section_editing",
        }
        for loop_num, loop_desc in loop_descriptions.items():
            key = f"review_loop{loop_num}"
            if result.get(key):
                loop_file = save_markdown_report(
                    result[key],
                    f"supervised_lit_review_loop{loop_num}",
                    title=f"Literature Review (After Loop {loop_num}: {loop_desc.replace('_', ' ').title()}): {topic}",
                    metadata={
                        "quality": quality,
                        "language": language,
                        "supervision": supervision_loops,
                        "loop": loop_num,
                        "loop_name": loop_desc,
                    },
                )
                logger.info(f"Loop {loop_num} review saved to: {loop_file}")

        # Save supervised review (v2, after supervision loops) as markdown
        if result.get("final_review_v2"):
            supervision_info = result.get("supervision", {})
            loops_run = supervision_info.get("loops_run", [])

            report_v2_file = save_markdown_report(
                result["final_review_v2"],
                "supervised_lit_review_v2",
                title=f"Literature Review (Supervised): {topic}",
                metadata={
                    "quality": quality,
                    "language": language,
                    "supervision": supervision_loops,
                    "loops_completed": ", ".join(loops_run) if loops_run else "none",
                },
            )
            logger.info(f"Supervised review (v2) saved to: {report_v2_file}")

        # Save human review items if any
        human_review = result.get("human_review_items", [])
        if human_review:
            review_file = save_json_result(human_review, "supervised_lit_review_human_items")
            logger.info(f"Human review items saved to: {review_file}")

        # Save PRISMA documentation
        if result.get("prisma_documentation"):
            prisma_file = save_markdown_report(
                result["prisma_documentation"],
                "supervised_lit_review_prisma",
            )
            logger.info(f"PRISMA docs saved to: {prisma_file}")

        return result, analysis

    except Exception as e:
        logger.error(f"Supervised literature review failed: {e}", exc_info=True)
        return {"error": str(e)}, {"error": str(e)}

    finally:
        # Clean up HTTP clients to avoid "Unclosed client session" warnings
        from core.utils.async_http_client import cleanup_all_clients
        await cleanup_all_clients()


if __name__ == "__main__":
    asyncio.run(main())
