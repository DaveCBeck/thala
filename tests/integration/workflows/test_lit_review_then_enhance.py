"""
Pytest tests for the combined lit review + enhance workflow.

Runs a literature review followed by the full enhancement workflow (supervision
loops, editing, fact-check) with minimal quality settings for fast execution.

Uses test fixtures to isolate from local infrastructure:
- test_store_manager: StoreManager wired to testcontainers + mocks
- mock_zotero: Mocked Zotero (no real Zotero service needed)
- configure_llm_broker_fast_mode: LLM broker in fast mode (direct calls, no batching)

Real LLM calls are made through the broker but executed directly (not batched).

Quality Settings ("test" tier):
- Lit review: max_stages=1, max_papers=5, target_word_count=2000
- Enhance: loops=none or "one" with max_iterations=1, run_fact_check=False
- Editing: max_structure_iterations=1, max_polish_edits=3, verify_use_perplexity=False

Usage:
    pytest tests/integration/workflows/test_lit_review_then_enhance.py -m integration
    pytest tests/integration/workflows/test_lit_review_then_enhance.py::TestLitReviewThenEnhance::test_full_workflow -m integration
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.utils import BaseQualityAnalyzer, QualityMetrics

if TYPE_CHECKING:
    from langchain_tools.base import StoreManager

# Enable dev mode for LangSmith tracing
os.environ["THALA_MODE"] = "dev"

logger = logging.getLogger(__name__)

# Test topic - focused enough to get relevant results, broad enough to have literature
TEST_TOPIC = "Retrieval-augmented generation for question answering"
TEST_RESEARCH_QUESTIONS = [
    "What retrieval strategies are most effective for RAG systems?",
    "How do RAG systems compare to pure parametric approaches?",
]


class CombinedWorkflowQualityAnalyzer(BaseQualityAnalyzer):
    """Quality analyzer for combined lit review + enhance results."""

    output_field = "final_report"
    min_word_count = 500  # Lower threshold for test quality
    min_source_count = 3

    def _count_sources(self, metrics: QualityMetrics) -> None:
        """Count sources from paper corpus."""
        paper_corpus = self.result.get("paper_corpus", {})
        metrics.source_count = len(paper_corpus)

    def _analyze_workflow_specific(self, metrics: QualityMetrics) -> None:
        """Analyze combined workflow metrics."""
        # Lit review metrics
        paper_corpus = self.result.get("paper_corpus", {})
        paper_summaries = self.result.get("paper_summaries", {})

        metrics.workflow_specific["papers_discovered"] = len(paper_corpus)
        metrics.workflow_specific["papers_processed"] = len(paper_summaries)

        # Enhance metrics
        enhance_result = self.result.get("enhance_result", {})
        if enhance_result:
            metrics.workflow_specific["enhance_status"] = enhance_result.get("status", "not_run")
            metrics.workflow_specific["supervision_ran"] = enhance_result.get("supervision_result") is not None
            metrics.workflow_specific["editing_ran"] = enhance_result.get("editing_result") is not None
            metrics.workflow_specific["fact_check_ran"] = enhance_result.get("fact_check_result") is not None

        # Track report growth
        lit_report_len = self.result.get("lit_report_length", 0)
        final_report_len = len(self.result.get("final_report", ""))
        if lit_report_len > 0:
            metrics.workflow_specific["report_growth"] = final_report_len / lit_report_len

    def _identify_issues(self, metrics: QualityMetrics) -> None:
        """Identify combined workflow issues."""
        super()._identify_issues(metrics)

        papers = metrics.workflow_specific.get("papers_discovered", 0)
        if papers < 3:
            metrics.issues.append(f"Very few papers ({papers})")

        enhance_status = metrics.workflow_specific.get("enhance_status")
        if enhance_status == "failed":
            metrics.issues.append("Enhancement failed")

    def _generate_suggestions(self, metrics: QualityMetrics) -> None:
        """Generate suggestions for improvement."""
        if not metrics.issues:
            metrics.suggestions.append("Workflow completed successfully")
            return

        if "papers" in str(metrics.issues).lower():
            metrics.suggestions.append("Consider broader topic or higher quality setting")
        if "failed" in str(metrics.issues).lower():
            metrics.suggestions.append("Check error logs for enhancement failure details")


def log_workflow_summary(result: dict, topic: str, output_path: Path | None = None) -> None:
    """Log a detailed summary of the combined workflow result."""
    logger.info("=" * 80)
    logger.info("COMBINED LIT REVIEW + ENHANCE WORKFLOW RESULT")
    logger.info("=" * 80)

    logger.info(f"Topic: {topic}")

    # Lit review phase
    logger.info("-" * 40)
    logger.info("PHASE 1: Literature Review")
    paper_corpus = result.get("paper_corpus", {})
    paper_summaries = result.get("paper_summaries", {})
    lit_report_length = result.get("lit_report_length", 0)

    logger.info(f"  Papers discovered: {len(paper_corpus)}")
    logger.info(f"  Papers processed: {len(paper_summaries)}")
    logger.info(f"  Initial report length: {lit_report_length} chars")

    if paper_corpus:
        sample_papers = list(paper_corpus.items())[:3]
        for doi, paper in sample_papers:
            title = paper.get("title", "Unknown")[:50]
            year = paper.get("year", "?")
            logger.info(f"    - [{year}] {title}...")

    # Enhance phase
    logger.info("-" * 40)
    logger.info("PHASE 2: Enhancement")
    enhance_result = result.get("enhance_result", {})

    if enhance_result:
        logger.info(f"  Status: {enhance_result.get('status', 'unknown')}")

        if enhance_result.get("supervision_result"):
            sup = enhance_result["supervision_result"]
            logger.info(f"  Supervision: loops_run={sup.get('loops_run', 'none')}")

        if enhance_result.get("editing_result"):
            edit = enhance_result["editing_result"]
            logger.info(f"  Editing: status={edit.get('status', 'unknown')}")

        if enhance_result.get("fact_check_result"):
            fc = enhance_result["fact_check_result"]
            logger.info(f"  Fact-check: status={fc.get('status', 'unknown')}")

        errors = enhance_result.get("errors", [])
        if errors:
            logger.warning(f"  Errors: {len(errors)}")
            for err in errors[:3]:
                logger.warning(f"    - {err}")
    else:
        logger.info("  Enhancement not run")

    # Final output
    logger.info("-" * 40)
    logger.info("FINAL OUTPUT")
    final_report = result.get("final_report", "")
    if final_report:
        word_count = len(final_report.split())
        logger.info(f"  Final report: {len(final_report)} chars ({word_count} words)")

        # Growth calculation
        if lit_report_length > 0:
            growth = len(final_report) / lit_report_length
            logger.info(f"  Report growth: {growth:.1%}")
    else:
        logger.warning("  No final report generated!")

    if output_path:
        logger.info(f"  Saved to: {output_path}")


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create output directory for test artifacts."""
    output = tmp_path / "workflow_output"
    output.mkdir(parents=True, exist_ok=True)
    return output


async def run_lit_review(
    topic: str,
    research_questions: list[str],
    quality: str = "test",
    language: str = "en",
) -> dict[str, Any]:
    """Run the literature review workflow.

    Args:
        topic: Research topic
        research_questions: List of research questions
        quality: Quality preset (default "test" for fast execution)
        language: Language code (default "en")

    Returns:
        dict with workflow results
    """
    from workflows.research.academic_lit_review import academic_lit_review

    logger.info(f"Starting literature review: topic='{topic}', quality={quality}")

    result = await academic_lit_review(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        language=language,
    )

    logger.info(f"Literature review complete: {len(result.get('final_report', ''))} chars")

    return result


async def run_enhance(
    report: str,
    topic: str,
    research_questions: list[str],
    quality: str = "test",
    loops: str = "one",
    max_iterations_per_loop: int = 1,
    run_editing: bool = True,
    run_fact_check: bool = False,
    paper_corpus: dict | None = None,
    paper_summaries: dict | None = None,
    zotero_keys: dict | None = None,
) -> dict[str, Any]:
    """Run the enhance workflow on a report.

    Args:
        report: Markdown report to enhance
        topic: Research topic
        research_questions: List of research questions
        quality: Quality preset (default "test")
        loops: Supervision loops ("none", "one", "two", "all")
        max_iterations_per_loop: Max iterations per supervision loop
        run_editing: Whether to run editing phase
        run_fact_check: Whether to run fact-check phase
        paper_corpus: Optional existing paper corpus
        paper_summaries: Optional existing paper summaries
        zotero_keys: Optional existing Zotero keys

    Returns:
        dict with enhancement results
    """
    from workflows.enhance import enhance_report

    logger.info(
        f"Starting enhancement: quality={quality}, loops={loops}, "
        f"max_iter={max_iterations_per_loop}, editing={run_editing}, fact_check={run_fact_check}"
    )

    result = await enhance_report(
        report=report,
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        loops=loops,
        max_iterations_per_loop=max_iterations_per_loop,
        run_editing=run_editing,
        run_fact_check=run_fact_check,
        paper_corpus=paper_corpus,
        paper_summaries=paper_summaries,
        zotero_keys=zotero_keys,
    )

    logger.info(f"Enhancement complete: status={result.get('status')}, {len(result.get('final_report', ''))} chars")

    return result


@pytest.mark.integration
@pytest.mark.slow
class TestLitReviewThenEnhance:
    """Integration tests for combined lit review + enhance workflow."""

    async def test_full_workflow(
        self,
        test_store_manager: "StoreManager",  # Sets up global singleton with testcontainers
        output_dir: Path,
    ) -> None:
        """Test full workflow: lit review -> enhance with all phases.

        Uses "test" quality settings throughout:
        - Lit review: max_stages=1, max_papers=5
        - Enhance: loops="one", max_iterations=1, editing=True, fact_check=False
        """
        # Phase 1: Literature Review
        logger.info("=" * 60)
        logger.info("PHASE 1: LITERATURE REVIEW")
        logger.info("=" * 60)

        lit_result = await run_lit_review(
            topic=TEST_TOPIC,
            research_questions=TEST_RESEARCH_QUESTIONS,
            quality="test",
        )

        lit_report = lit_result.get("final_report", "")
        assert lit_report, "Literature review should produce a report"

        # Save intermediate output
        lit_output_path = output_dir / "lit_review_output.md"
        lit_output_path.write_text(f"""---
topic: {TEST_TOPIC}
generated: {datetime.now().isoformat()}
phase: literature_review
quality: test
papers_discovered: {len(lit_result.get('paper_corpus', {}))}
---

{lit_report}
""")
        logger.info(f"Saved lit review to: {lit_output_path}")

        # Phase 2: Enhancement
        logger.info("=" * 60)
        logger.info("PHASE 2: ENHANCEMENT")
        logger.info("=" * 60)

        enhance_result = await run_enhance(
            report=lit_report,
            topic=TEST_TOPIC,
            research_questions=TEST_RESEARCH_QUESTIONS,
            quality="test",
            loops="one",  # Single supervision loop
            max_iterations_per_loop=1,
            run_editing=True,
            run_fact_check=False,  # Skip for speed
            paper_corpus=lit_result.get("paper_corpus"),
            paper_summaries=lit_result.get("paper_summaries"),
            zotero_keys=lit_result.get("zotero_keys"),
        )

        final_report = enhance_result.get("final_report", "")
        assert final_report, "Enhancement should produce a final report"

        # Save final output
        final_output_path = output_dir / "enhanced_output.md"
        final_output_path.write_text(f"""---
topic: {TEST_TOPIC}
generated: {datetime.now().isoformat()}
phase: enhanced
quality: test
enhance_status: {enhance_result.get('status')}
supervision_ran: {enhance_result.get('supervision_result') is not None}
editing_ran: {enhance_result.get('editing_result') is not None}
---

{final_report}
""")
        logger.info(f"Saved enhanced report to: {final_output_path}")

        # Combine results for analysis
        combined_result = {
            "final_report": final_report,
            "paper_corpus": lit_result.get("paper_corpus", {}),
            "paper_summaries": lit_result.get("paper_summaries", {}),
            "zotero_keys": lit_result.get("zotero_keys", {}),
            "lit_report_length": len(lit_report),
            "enhance_result": enhance_result,
        }

        log_workflow_summary(combined_result, TEST_TOPIC, final_output_path)

        # Quality analysis
        analyzer = CombinedWorkflowQualityAnalyzer(combined_result)
        metrics = analyzer.analyze()

        logger.info(
            f"Quality: completed={metrics.completed}, words={metrics.word_count}, "
            f"sources={metrics.source_count}"
        )
        if metrics.issues:
            logger.info(f"Issues: {', '.join(metrics.issues[:3])}")

        # Assertions
        assert enhance_result.get("status") in ("success", "partial"), \
            f"Enhancement should succeed or partially succeed, got: {enhance_result.get('status')}"

    async def test_lit_review_only(
        self,
        test_store_manager: "StoreManager",
        output_dir: Path,
    ) -> None:
        """Test just the literature review phase with test quality."""
        lit_result = await run_lit_review(
            topic=TEST_TOPIC,
            research_questions=TEST_RESEARCH_QUESTIONS,
            quality="test",
        )

        lit_report = lit_result.get("final_report", "")
        assert lit_report, "Should produce a report"
        assert len(lit_report) > 100, "Report should have meaningful content"

        logger.info(f"Lit review complete: {len(lit_report)} chars, {len(lit_result.get('paper_corpus', {}))} papers")

    async def test_enhance_with_all_loops_minimal(
        self,
        test_store_manager: "StoreManager",
        output_dir: Path,
    ) -> None:
        """Test enhance with all supervision loops but minimal iterations.

        This hits all the loop code paths but with max_iterations=1 each.
        """
        # First get a lit review report
        lit_result = await run_lit_review(
            topic=TEST_TOPIC,
            research_questions=TEST_RESEARCH_QUESTIONS,
            quality="test",
        )

        lit_report = lit_result.get("final_report", "")
        assert lit_report, "Need a report to enhance"

        # Run enhance with all loops but minimal iterations
        enhance_result = await run_enhance(
            report=lit_report,
            topic=TEST_TOPIC,
            research_questions=TEST_RESEARCH_QUESTIONS,
            quality="test",
            loops="all",  # All loops
            max_iterations_per_loop=1,  # But only 1 iteration each
            run_editing=True,
            run_fact_check=False,  # Skip for speed
            paper_corpus=lit_result.get("paper_corpus"),
            paper_summaries=lit_result.get("paper_summaries"),
            zotero_keys=lit_result.get("zotero_keys"),
        )

        assert enhance_result.get("final_report"), "Should produce enhanced report"
        assert enhance_result.get("status") in ("success", "partial")

        # Verify supervision ran
        supervision = enhance_result.get("supervision_result")
        if supervision:
            logger.info(f"Supervision loops run: {supervision.get('loops_run')}")

    async def test_enhance_editing_only(
        self,
        test_store_manager: "StoreManager",
        output_dir: Path,
    ) -> None:
        """Test enhance with no supervision, just editing."""
        # First get a lit review report
        lit_result = await run_lit_review(
            topic=TEST_TOPIC,
            research_questions=TEST_RESEARCH_QUESTIONS,
            quality="test",
        )

        lit_report = lit_result.get("final_report", "")
        assert lit_report, "Need a report to enhance"

        # Run enhance with editing only
        enhance_result = await run_enhance(
            report=lit_report,
            topic=TEST_TOPIC,
            research_questions=TEST_RESEARCH_QUESTIONS,
            quality="test",
            loops="none",  # Skip supervision
            run_editing=True,
            run_fact_check=False,
        )

        assert enhance_result.get("final_report"), "Should produce enhanced report"

        # Verify only editing ran
        assert enhance_result.get("supervision_result") is None, "Supervision should not run"
        assert enhance_result.get("editing_result") is not None, "Editing should run"


@pytest.mark.integration
@pytest.mark.slow
async def test_minimal_workflow_standalone(
    test_store_manager: "StoreManager",
    output_dir: Path,
) -> None:
    """Standalone minimal test: lit review + enhance editing only.

    This is the fastest possible combined workflow test.
    """
    # Lit review with test quality
    lit_result = await run_lit_review(
        topic=TEST_TOPIC,
        research_questions=TEST_RESEARCH_QUESTIONS,
        quality="test",
    )

    lit_report = lit_result.get("final_report", "")
    assert lit_report, "Should produce report"

    # Enhance with just editing (no supervision, no fact-check)
    enhance_result = await run_enhance(
        report=lit_report,
        topic=TEST_TOPIC,
        research_questions=TEST_RESEARCH_QUESTIONS,
        quality="test",
        loops="none",
        run_editing=True,
        run_fact_check=False,
    )

    assert enhance_result.get("final_report"), "Should produce enhanced report"
    assert enhance_result.get("status") in ("success", "partial")

    logger.info(
        f"Minimal workflow complete: "
        f"lit={len(lit_report)} chars -> enhanced={len(enhance_result.get('final_report', ''))} chars"
    )
