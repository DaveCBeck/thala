"""
Pytest tests for the academic literature review workflow.

Runs comprehensive literature review with real LLM calls but isolated
from local infrastructure using testcontainers and mocks.

Uses test fixtures to isolate from local infrastructure:
- test_store_manager: StoreManager wired to testcontainers + mocks
- mock_zotero: Mocked Zotero (no real Zotero service needed)
- configure_llm_broker_fast_mode: LLM broker in fast mode (direct calls, no batching)

Real LLM calls are made through the broker but executed directly (not batched).

Usage:
    pytest tests/integration/workflows/test_academic_lit_review.py -m integration --quality quick
    pytest tests/integration/workflows/test_academic_lit_review.py -m integration --quality standard --language es
"""

import logging
import os
from typing import TYPE_CHECKING

import pytest

from tests.utils import (
    BaseQualityAnalyzer,
    QualityMetrics,
)
from workflows.shared.workflow_state_store import load_workflow_state

if TYPE_CHECKING:
    from langchain_tools.base import StoreManager

# Enable dev mode for LangSmith tracing
os.environ["THALA_MODE"] = "dev"

logger = logging.getLogger(__name__)


class AcademicLitReviewQualityAnalyzer(BaseQualityAnalyzer):
    """Quality analyzer for academic literature review results."""

    output_field = "final_report"
    min_word_count = 1000
    min_source_count = 5

    def _count_sources(self, metrics: QualityMetrics) -> None:
        """Count sources from paper corpus and references."""
        run_id = self.result.get("langsmith_run_id")
        if not run_id:
            return

        state = load_workflow_state("academic_lit_review", run_id)
        paper_corpus = state.get("paper_corpus", {}) if state else {}
        references = state.get("references", []) if state else []
        metrics.source_count = max(len(paper_corpus), len(references))

    def _analyze_workflow_specific(self, metrics: QualityMetrics) -> None:
        """Analyze academic lit review specific metrics."""
        run_id = self.result.get("langsmith_run_id")
        if not run_id:
            return

        state = load_workflow_state("academic_lit_review", run_id)
        if not state:
            return

        # Paper metrics
        paper_corpus = state.get("paper_corpus", {})
        paper_summaries = state.get("paper_summaries", {})

        metrics.workflow_specific["papers_discovered"] = len(paper_corpus)
        metrics.workflow_specific["papers_processed"] = len(paper_summaries)

        if paper_corpus:
            metrics.workflow_specific["processing_rate"] = len(paper_summaries) / len(
                paper_corpus
            )

        # Clustering metrics
        clusters = state.get("clusters", [])
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
        diffusion = state.get("diffusion", {})
        if diffusion:
            metrics.workflow_specific["diffusion_stages"] = diffusion.get(
                "current_stage", 0
            )
            metrics.workflow_specific["saturation_reached"] = diffusion.get(
                "is_saturated", False
            )

        # References
        references = state.get("references", [])
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


def log_result_summary(result: dict, topic: str) -> None:
    """Log a detailed summary of the literature review result."""
    logger.info("=" * 80)
    logger.info("ACADEMIC LITERATURE REVIEW RESULT")
    logger.info("=" * 80)

    logger.info(f"Topic: {topic}")

    # Load state for detailed metrics
    run_id = result.get("langsmith_run_id")
    state = load_workflow_state("academic_lit_review", run_id) if run_id else None

    # Paper Corpus
    paper_corpus = state.get("paper_corpus", {}) if state else {}
    logger.info(f"Papers discovered: {len(paper_corpus)}")

    if paper_corpus:
        sample_papers = list(paper_corpus.items())[:3]
        for doi, paper in sample_papers:
            title = paper.get("title", "Unknown")[:50]
            year = paper.get("year", "?")
            logger.info(f"  - [{year}] {title}...")

    # Paper Summaries
    paper_summaries = state.get("paper_summaries", {}) if state else {}
    logger.info(f"Papers processed: {len(paper_summaries)}")

    # Clusters
    clusters = state.get("clusters", []) if state else []
    logger.info(f"Thematic clusters: {len(clusters)}")

    # Final Report
    final_report = result.get("final_report", "")
    if final_report:
        word_count = len(final_report.split())
        logger.info(f"Final report: {len(final_report)} chars ({word_count} words)")

    # References
    references = state.get("references", []) if state else []
    logger.info(f"References: {len(references)}")

    # Storage and Tracing
    if run_id:
        logger.info(f"LangSmith Run ID: {run_id}")

    errors = result.get("errors", [])
    if errors:
        logger.error(f"Errors: {len(errors)}")
        for error in errors[:3]:
            logger.error(f"  - {error}")


@pytest.mark.integration
@pytest.mark.slow
class TestAcademicLitReview:
    """Integration tests for academic literature review workflow."""

    async def test_quick_literature_review(
        self,
        test_store_manager: "StoreManager",  # Sets up global singleton with testcontainers
        quality_level: str,
        language: str,
    ) -> None:
        """Test quick literature review on a topic."""
        from workflows.research.academic_lit_review import academic_lit_review

        # test_store_manager fixture configures the global StoreManager singleton
        topic = "The impact of large language models on software engineering practices"
        research_questions = [
            f"What are the main research themes in {topic}?",
            f"What methodological approaches are used to study {topic}?",
        ]

        # Use 'test' quality for CI/quick runs, or the configured quality level
        effective_quality = quality_level if quality_level != "quick" else "test"

        logger.info(f"Starting literature review on: {topic}")
        logger.info(f"Quality: {effective_quality}, Language: {language}")

        # The test_store_manager fixture sets up the global singleton
        # with testcontainers and mocks, so workflows use it automatically
        result = await academic_lit_review(
            topic=topic,
            research_questions=research_questions,
            quality=effective_quality,
            language=language,
        )

        log_result_summary(result, topic)

        # Assertions
        assert result.get("final_report"), "Should have final report"
        assert len(result.get("final_report", "")) > 100, "Report should have content"

        # Quality analysis (informational, not strict assertions for test quality)
        analyzer = AcademicLitReviewQualityAnalyzer(result)
        metrics = analyzer.analyze()

        logger.info(f"Quality: completed={metrics.completed}, words={metrics.word_count}, sources={metrics.source_count}")
        if metrics.issues:
            logger.info(f"Issues: {', '.join(metrics.issues[:3])}")

    async def test_literature_review_with_date_range(
        self,
        test_store_manager: "StoreManager",  # Sets up global singleton with testcontainers
    ) -> None:
        """Test literature review with date range constraint."""
        from workflows.research.academic_lit_review import academic_lit_review

        # test_store_manager fixture configures the global StoreManager singleton
        topic = "transformer architectures in NLP"
        date_range = (2020, 2024)

        logger.info(f"Starting literature review on: {topic}")
        logger.info(f"Date range: {date_range}")

        result = await academic_lit_review(
            topic=topic,
            research_questions=[f"What are the key advances in {topic}?"],
            quality="test",
            date_range=date_range,
        )

        log_result_summary(result, topic)

        # Assertions
        assert result.get("final_report"), "Should have final report"
