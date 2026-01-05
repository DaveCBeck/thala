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

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable dev mode for LangSmith tracing before any imports
os.environ["THALA_MODE"] = "dev"

# Setup logging - both console and file
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Create a unique log file for each run
_log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = LOG_DIR / f"lit_review_{_log_timestamp}.log"

# Configure root logger for both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(LOG_FILE, mode='w'),  # File output
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {LOG_FILE}")

# Output directory for results
OUTPUT_DIR = Path(__file__).parent / "test_data"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

VALID_QUALITIES = ["test", "quick", "standard", "comprehensive", "high_quality"]
DEFAULT_QUALITY = "quick"


# =============================================================================
# Checkpoint Utilities
# =============================================================================


def save_checkpoint(state: dict, name: str) -> Path:
    """Save workflow state to a checkpoint file for later resumption."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{name}.json"
    with open(checkpoint_file, "w") as f:
        json.dump(state, f, indent=2, default=str)
    logger.info(f"Checkpoint saved: {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(name: str) -> dict | None:
    """Load workflow state from a checkpoint file."""
    checkpoint_file = CHECKPOINT_DIR / f"{name}.json"
    if not checkpoint_file.exists():
        logger.error(f"Checkpoint not found: {checkpoint_file}")
        return None
    with open(checkpoint_file, "r") as f:
        state = json.load(f)
    logger.info(f"Checkpoint loaded: {checkpoint_file}")
    return state


def print_result_summary(result: dict, topic: str) -> None:
    """Print a detailed summary of the literature review result."""
    print("\n" + "=" * 80)
    print("ACADEMIC LITERATURE REVIEW RESULT")
    print("=" * 80)

    # Topic
    print(f"\nTopic: {topic}")

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

    # Paper Corpus
    paper_corpus = result.get("paper_corpus", {})
    print(f"\n--- Paper Corpus ---")
    print(f"Total papers discovered: {len(paper_corpus)}")

    if paper_corpus:
        # Sample papers
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

    # Final Review
    final_review = result.get("final_review", "")
    if final_review:
        print(f"\n--- Final Review ---")
        word_count = len(final_review.split())
        print(f"Length: {len(final_review)} chars ({word_count} words)")
        # Show first 1500 chars
        preview = final_review[:1500]
        if len(final_review) > 1500:
            preview += "\n\n... [truncated] ..."
        print(preview)

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
        print(prisma[:500])
        if len(prisma) > 500:
            print("... [truncated]")

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
    errors = result.get("errors", [])
    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for err in errors:
            phase = err.get("phase", "unknown")
            error = err.get("error", "unknown")
            print(f"  [{phase}]: {error}")

    print("\n" + "=" * 80)


def analyze_quality(result: dict) -> dict:
    """Analyze literature review quality and return metrics."""
    analysis = {
        "metrics": {},
        "issues": [],
        "suggestions": [],
    }

    # Basic completion metrics
    final_review = result.get("final_review", "")
    if final_review and not final_review.startswith("Literature review generation failed"):
        word_count = len(final_review.split())
        analysis["metrics"]["review_word_count"] = word_count
        analysis["metrics"]["review_char_count"] = len(final_review)
        analysis["metrics"]["completed"] = True

        if word_count < 1000:
            analysis["issues"].append(f"Review is short ({word_count} words)")
    else:
        analysis["metrics"]["completed"] = False
        analysis["issues"].append("Review generation failed or incomplete")

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

        # Check for cluster quality
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
        analysis["suggestions"].append("Literature review appears comprehensive")
    else:
        if "Low paper count" in str(analysis["issues"]):
            analysis["suggestions"].append("Consider broadening search terms or using higher quality setting")
        if "Few clusters" in str(analysis["issues"]):
            analysis["suggestions"].append("More papers needed for meaningful clustering")
        if "short" in str(analysis["issues"]).lower():
            analysis["suggestions"].append("Use higher quality setting for longer reviews")

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


async def run_literature_review(
    topic: str,
    research_questions: list[str],
    quality: str = "quick",
    date_range: tuple[int, int] | None = None,
    language: str = "en",
) -> dict:
    """Run the academic literature review workflow on a topic."""
    from workflows.research.subgraphs.academic_lit_review import academic_lit_review

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

    return result


# =============================================================================
# Checkpointed Run Functions
# =============================================================================


def build_initial_state(
    topic: str,
    research_questions: list[str],
    quality: str,
    date_range: tuple[int, int] | None,
    language: str = "en",
) -> dict:
    """Build initial state for the literature review workflow.

    This replicates the state initialization from academic_lit_review() in graph.py
    to allow running phases individually with checkpointing.
    """
    import uuid
    from workflows.research.subgraphs.academic_lit_review.state import (
        QUALITY_PRESETS,
        LitReviewInput,
        LitReviewDiffusionState,
    )
    from workflows.shared.language import get_language_config

    if quality not in QUALITY_PRESETS:
        logger.warning(f"Unknown quality '{quality}', using 'standard'")
        quality = "standard"

    quality_settings = dict(QUALITY_PRESETS[quality])
    language_config = get_language_config(language)

    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        date_range=date_range,
        include_books=True,
        focus_areas=None,
        exclude_terms=None,
        max_papers=None,
        language_code=language,
    )

    return {
        "input": input_data,
        "quality_settings": quality_settings,
        "language_config": language_config,
        "keyword_papers": [],
        "citation_papers": [],
        "expert_papers": [],
        "book_dois": [],
        "diffusion": LitReviewDiffusionState(
            current_stage=0,
            max_stages=quality_settings["max_stages"],
            stages=[],
            saturation_threshold=quality_settings["saturation_threshold"],
            is_saturated=False,
            consecutive_low_coverage=0,
            total_papers_discovered=0,
            total_papers_relevant=0,
            total_papers_rejected=0,
        ),
        "paper_corpus": {},
        "paper_summaries": {},
        "citation_edges": [],
        "paper_nodes": {},
        "papers_to_process": [],
        "papers_processed": [],
        "papers_failed": [],
        "bertopic_clusters": None,
        "llm_topic_schema": None,
        "clusters": [],
        "section_drafts": {},
        "final_review": None,
        "references": [],
        "prisma_documentation": None,
        "elasticsearch_ids": {},
        "zotero_keys": {},
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "current_phase": "discovery",
        "current_status": "Starting literature review",
        "langsmith_run_id": str(uuid.uuid4()),
        "errors": [],
    }


async def run_with_checkpoints(
    topic: str,
    research_questions: list[str],
    quality: str = "quick",
    date_range: tuple[int, int] | None = None,
    checkpoint_prefix: str = "latest",
    language: str = "en",
) -> dict:
    """Run full workflow with automatic checkpoint saves after expensive phases.

    Saves checkpoints:
    - {prefix}_after_diffusion: After paper corpus is complete
    - {prefix}_after_processing: After paper summaries are complete
    """
    from workflows.research.subgraphs.academic_lit_review.graph import (
        discovery_phase_node,
        diffusion_phase_node,
        processing_phase_node,
        clustering_phase_node,
        synthesis_phase_node,
        supervision_phase_node,
    )

    logger.info(f"Starting checkpointed literature review on: {topic}")
    logger.info(f"Checkpoint prefix: {checkpoint_prefix}")
    logger.info(f"Language: {language}")

    # Build initial state
    state = build_initial_state(topic, research_questions, quality, date_range, language)

    # Phase 1: Discovery
    logger.info("Running discovery phase...")
    state.update(await discovery_phase_node(state))

    # Phase 2: Diffusion (expensive - API calls, relevance scoring)
    logger.info("Running diffusion phase...")
    state.update(await diffusion_phase_node(state))
    save_checkpoint(state, f"{checkpoint_prefix}_after_diffusion")

    # Phase 3: Processing (expensive - PDF download, Marker, LLM summaries)
    logger.info("Running processing phase...")
    state.update(await processing_phase_node(state))
    save_checkpoint(state, f"{checkpoint_prefix}_after_processing")

    # Phase 4: Clustering
    logger.info("Running clustering phase...")
    state.update(await clustering_phase_node(state))

    # Phase 5: Synthesis
    logger.info("Running synthesis phase...")
    state.update(await synthesis_phase_node(state))

    # Phase 6: Supervision (iterative improvement)
    logger.info("Running supervision phase...")
    state.update(await supervision_phase_node(state))

    return state


async def run_from_diffusion_checkpoint(checkpoint_prefix: str) -> dict:
    """Resume workflow from after-diffusion checkpoint.

    Runs: processing -> clustering -> synthesis -> supervision
    Skips: discovery, diffusion
    """
    from workflows.research.subgraphs.academic_lit_review.graph import (
        processing_phase_node,
        clustering_phase_node,
        synthesis_phase_node,
        supervision_phase_node,
    )

    checkpoint_name = f"{checkpoint_prefix}_after_diffusion"
    state = load_checkpoint(checkpoint_name)
    if not state:
        raise ValueError(f"Checkpoint not found: {checkpoint_name}")

    logger.info(f"Resuming from diffusion checkpoint: {checkpoint_name}")
    logger.info(f"Paper corpus size: {len(state.get('paper_corpus', {}))}")

    # Phase 3: Processing
    logger.info("Running processing phase...")
    state.update(await processing_phase_node(state))
    save_checkpoint(state, f"{checkpoint_prefix}_after_processing")

    # Phase 4: Clustering
    logger.info("Running clustering phase...")
    state.update(await clustering_phase_node(state))

    # Phase 5: Synthesis
    logger.info("Running synthesis phase...")
    state.update(await synthesis_phase_node(state))

    # Phase 6: Supervision
    logger.info("Running supervision phase...")
    state.update(await supervision_phase_node(state))

    return state


async def run_from_processing_checkpoint(checkpoint_prefix: str) -> dict:
    """Resume workflow from after-processing checkpoint.

    Runs: clustering -> synthesis -> supervision
    Skips: discovery, diffusion, processing
    """
    from workflows.research.subgraphs.academic_lit_review.graph import (
        clustering_phase_node,
        synthesis_phase_node,
        supervision_phase_node,
    )

    checkpoint_name = f"{checkpoint_prefix}_after_processing"
    state = load_checkpoint(checkpoint_name)
    if not state:
        raise ValueError(f"Checkpoint not found: {checkpoint_name}")

    logger.info(f"Resuming from processing checkpoint: {checkpoint_name}")
    logger.info(f"Paper summaries size: {len(state.get('paper_summaries', {}))}")

    # Phase 4: Clustering
    logger.info("Running clustering phase...")
    state.update(await clustering_phase_node(state))

    # Phase 5: Synthesis
    logger.info("Running synthesis phase...")
    state.update(await synthesis_phase_node(state))

    # Phase 6: Supervision
    logger.info("Running supervision phase...")
    state.update(await supervision_phase_node(state))

    return state


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run academic literature review workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "transformer architectures"              # Quick review with checkpoints
  %(prog)s "transformer architectures" standard     # Standard review with checkpoints
  %(prog)s "AI in drug discovery" comprehensive     # Comprehensive review

Checkpoint examples:
  %(prog)s "topic" quick --checkpoint-prefix mytest           # Full run, saves checkpoints
  %(prog)s --resume-from processing --checkpoint-prefix mytest  # Resume from processing
  %(prog)s --resume-from diffusion --checkpoint-prefix mytest   # Resume from diffusion
  %(prog)s "topic" quick --no-checkpoint                      # Original behavior (no checkpoints)
        """
    )

    parser.add_argument(
        "topic",
        nargs="?",
        default="The impact of large language models on software engineering practices",
        help="Research topic for literature review"
    )
    parser.add_argument(
        "quality",
        nargs="?",
        default=DEFAULT_QUALITY,
        choices=VALID_QUALITIES,
        help=f"Quality level (default: {DEFAULT_QUALITY})"
    )
    parser.add_argument(
        "--questions", "-q",
        type=str,
        nargs="+",
        default=None,
        help="Research questions (if not provided, will be auto-generated)"
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=None,
        help="Start year for date filter"
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=None,
        help="End year for date filter"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        choices=["diffusion", "processing"],
        default=None,
        help="Resume from a checkpoint (skips earlier phases)"
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="latest",
        help="Prefix for checkpoint files (default: 'latest')"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable automatic checkpointing (use original workflow)"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Language code for the review (default: en). Supported: en, es, zh, ja, de, fr, pt, ko, ru, ar, it, nl, pl, tr, vi, th, id, hi, bn, sv, no, da, fi, cs, el, he, uk, ro, hu"
    )

    return parser.parse_args()


async def main():
    """Run academic literature review test."""
    args = parse_args()

    topic = args.topic
    quality = args.quality
    checkpoint_prefix = args.checkpoint_prefix
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

    # Determine run mode
    if args.resume_from:
        mode = f"resume from {args.resume_from}"
    elif args.no_checkpoint:
        mode = "no checkpoints"
    else:
        mode = "with checkpoints"

    print(f"\n{'=' * 80}")
    print("ACADEMIC LITERATURE REVIEW TEST")
    print(f"{'=' * 80}")
    print(f"\nTopic: {topic}")
    print(f"Quality: {quality}")
    print(f"Language: {language}")
    print(f"Mode: {mode}")
    if not args.resume_from:
        print(f"Research Questions:")
        for q in research_questions:
            print(f"  - {q}")
    if date_range:
        print(f"Date Range: {date_range[0]}-{date_range[1]}")
    if not args.no_checkpoint:
        print(f"Checkpoint prefix: {checkpoint_prefix}")
    print(f"LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'thala-dev')}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        # Choose run function based on mode
        if args.resume_from == "diffusion":
            result = await run_from_diffusion_checkpoint(checkpoint_prefix)
            # Extract topic from checkpoint for display
            topic = result.get("input", {}).get("topic", topic)
        elif args.resume_from == "processing":
            result = await run_from_processing_checkpoint(checkpoint_prefix)
            # Extract topic from checkpoint for display
            topic = result.get("input", {}).get("topic", topic)
        elif args.no_checkpoint:
            result = await run_literature_review(
                topic=topic,
                research_questions=research_questions,
                quality=quality,
                date_range=date_range,
                language=language,
            )
        else:
            result = await run_with_checkpoints(
                topic=topic,
                research_questions=research_questions,
                quality=quality,
                date_range=date_range,
                checkpoint_prefix=checkpoint_prefix,
                language=language,
            )

        # Print detailed result summary
        print_result_summary(result, topic)

        # Analyze quality
        analysis = analyze_quality(result)
        print_quality_analysis(analysis)

        # Save results
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save full result
        result_file = OUTPUT_DIR / f"lit_review_result_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Full result saved to: {result_file}")

        # Save analysis
        analysis_file = OUTPUT_DIR / f"lit_review_analysis_{timestamp}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to: {analysis_file}")

        # Save final review (v1, before supervision) as markdown
        if result.get("final_review"):
            report_file = OUTPUT_DIR / f"lit_review_{timestamp}.md"
            with open(report_file, "w") as f:
                f.write(f"# Literature Review: {topic}\n\n")
                f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write(f"*Quality: {quality} | Language: {language}*\n\n")
                f.write("---\n\n")
                f.write(result["final_review"])
            logger.info(f"Review (v1) saved to: {report_file}")

        # Save supervised review (v2, after supervision loop) as markdown
        if result.get("final_review_v2"):
            report_v2_file = OUTPUT_DIR / f"lit_review_v2_{timestamp}.md"
            with open(report_v2_file, "w") as f:
                f.write(f"# Literature Review (Supervised): {topic}\n\n")
                f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write(f"*Quality: {quality} | Language: {language}*\n\n")
                f.write("---\n\n")
                f.write(result["final_review_v2"])
            logger.info(f"Review (v2) saved to: {report_v2_file}")

        # Save PRISMA documentation
        if result.get("prisma_documentation"):
            prisma_file = OUTPUT_DIR / f"lit_review_prisma_{timestamp}.md"
            with open(prisma_file, "w") as f:
                f.write(result["prisma_documentation"])
            logger.info(f"PRISMA docs saved to: {prisma_file}")

        return result, analysis

    except Exception as e:
        logger.error(f"Literature review failed: {e}", exc_info=True)
        return {"error": str(e)}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
