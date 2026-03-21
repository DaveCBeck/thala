"""
Integration test for the web-augmented literature review workflow.

Runs the full workflow with quality="test" settings to exercise:
- Web scan phase (Perplexity searches + LLM synthesis)
- Parallel research (academic lit review + deep_research with recency_filter)
- Combine phase (Opus merge of academic + web reports)
- Enhancement (supervision + editing)

Skips evening_reads and save_and_spawn to keep the test focused on the
new phases. Uses test_store_manager for isolated store access.

Usage:
    pytest tests/integration/workflows/test_lit_review_web_augmented.py -m integration -v
"""

import logging
import os
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from langchain_tools.base import StoreManager

os.environ["THALA_MODE"] = "dev"

logger = logging.getLogger(__name__)

TEST_TOPIC = "Brain-computer interfaces for motor rehabilitation"
TEST_RESEARCH_QUESTIONS = [
    "What non-invasive BCI approaches show the most promise for motor recovery?",
    "How do closed-loop BCI systems compare to open-loop for rehabilitation outcomes?",
]


@pytest.mark.integration
@pytest.mark.slow
class TestLitReviewWebAugmented:
    """Integration tests for the web-augmented literature review workflow."""

    async def test_web_scan_phase_standalone(
        self,
        test_store_manager: "StoreManager",
    ) -> None:
        """Test web scan phase in isolation — fastest possible integration check."""
        from core.task_queue.workflows.lit_review_web_augmented.phases.web_scan import (
            run_web_scan_phase,
        )

        result = await run_web_scan_phase(
            topic=TEST_TOPIC,
            research_questions=TEST_RESEARCH_QUESTIONS,
            web_scan_window_days=30,
        )

        # Should produce augmented questions
        augmented = result["augmented_research_questions"]
        assert len(augmented) >= len(TEST_RESEARCH_QUESTIONS), (
            f"Should preserve at least original questions, got {len(augmented)}"
        )

        # Should preserve originals
        assert result["original_research_questions"] == TEST_RESEARCH_QUESTIONS

        # Should have raw results from Perplexity
        assert len(result["raw_results"]) > 0, "Should have at least some search results"

        # Landscape should be a non-empty string
        assert isinstance(result["recent_landscape"], str)

        logger.info(
            f"Web scan complete: {len(augmented)} augmented questions, "
            f"{len(result['raw_results'])} raw results, "
            f"landscape={len(result['recent_landscape'])} chars"
        )

    async def test_full_workflow(
        self,
        test_store_manager: "StoreManager",
    ) -> None:
        """Test full web-augmented workflow end-to-end with quality=test.

        Exercises: web_scan → parallel(lit+web) → combine → supervision → editing.
        Stops before evening_reads/save_and_spawn (those are tested elsewhere and
        would add significant runtime).
        """
        from core.task_queue.workflows.lit_review_web_augmented import (
            LitReviewWebAugmentedWorkflow,
        )

        workflow = LitReviewWebAugmentedWorkflow()

        # Build a minimal task dict
        task: dict[str, Any] = {
            "id": "test-web-augmented-001",
            "task_type": "lit_review_web_augmented",
            "topic": TEST_TOPIC,
            "research_questions": TEST_RESEARCH_QUESTIONS,
            "category": "science",
            "priority": 2,
            "status": "in_progress",
            "quality": "test",
            "language": "en",
            "date_range": None,
            "web_scan_window_days": 30,
        }

        # Track checkpoints
        checkpoints: list[str] = []

        def checkpoint_callback(phase: str, **kwargs: Any) -> None:
            checkpoints.append(phase)
            logger.info(f"Checkpoint: {phase}")

        result = await workflow.run(task, checkpoint_callback)

        # --- Assertions ---
        status = result.get("status")
        assert status in ("success", "partial"), f"Expected success/partial, got {status}. Errors: {result.get('errors')}"

        # Should have a combined report
        final_report = result.get("final_report")
        assert final_report, "Should produce a final report"
        assert len(final_report) > 200, f"Report too short: {len(final_report)} chars"

        # Lit review leg should have run
        lit_review = result.get("lit_review")
        if lit_review:
            assert lit_review.get("final_report"), "Lit review should produce a report"
            logger.info(f"Lit review: {len(lit_review.get('paper_corpus', {}))} papers")

        # Web research leg should have run
        web_research = result.get("web_research")
        if web_research:
            assert web_research.get("final_report"), "Web research should produce a report"
            logger.info(f"Web research: {web_research.get('source_count', 0)} sources")

        # Combined result should pass through academic metadata when present
        combined = result.get("combined")
        if combined and lit_review:
            # paper_corpus/zotero_keys may be None at test quality — just verify the keys exist
            assert "paper_corpus" in combined, "Combined should include paper_corpus key"
            assert "zotero_keys" in combined, "Combined should include zotero_keys key"

        # Checkpoints should include all expected phases
        assert "web_scan" in checkpoints
        assert "parallel_research" in checkpoints

        # Log summary
        word_count = len(final_report.split()) if final_report else 0
        logger.info(
            f"Web-augmented workflow complete: status={status}, "
            f"report={word_count} words, "
            f"phases={checkpoints}, "
            f"errors={result.get('errors', [])}"
        )
