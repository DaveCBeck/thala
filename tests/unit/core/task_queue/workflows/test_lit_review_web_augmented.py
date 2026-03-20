"""Tests for the web-augmented literature review workflow."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_tools.perplexity import _iso_to_perplexity_date


# ---------------------------------------------------------------------------
# Perplexity date filtering
# ---------------------------------------------------------------------------


class TestPerplexityDateConversion:
    def test_basic_conversion(self):
        assert _iso_to_perplexity_date("2026-01-15") == "1/15/2026"

    def test_strips_leading_zeros(self):
        assert _iso_to_perplexity_date("2026-03-05") == "3/5/2026"

    def test_december(self):
        assert _iso_to_perplexity_date("2025-12-31") == "12/31/2025"


class TestPerplexitySearchDateParams:
    """Verify date params are added to the API payload."""

    @pytest.mark.asyncio
    async def test_after_date_in_payload(self):
        """after_date should add search_after_date_filter to payload."""
        from langchain_tools.perplexity import perplexity_search

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"results": []}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("langchain_tools.perplexity._get_perplexity", return_value=mock_client):
            await perplexity_search.ainvoke(
                {
                    "query": "test query",
                    "limit": 5,
                    "after_date": "2026-02-01",
                }
            )

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["search_after_date_filter"] == "2/1/2026"

    @pytest.mark.asyncio
    async def test_no_date_params_by_default(self):
        """Without date params, payload should not contain date filters."""
        from langchain_tools.perplexity import perplexity_search

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"results": []}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("langchain_tools.perplexity._get_perplexity", return_value=mock_client):
            await perplexity_search.ainvoke({"query": "test query", "limit": 5})

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "search_after_date_filter" not in payload
        assert "search_before_date_filter" not in payload


# ---------------------------------------------------------------------------
# Recency filter quota logic
# ---------------------------------------------------------------------------


class TestRecencyFilterQuota:
    """Test that date filtering is applied to the correct fraction of queries."""

    @pytest.mark.asyncio
    async def test_quota_applies_to_fraction(self):
        """With quota=0.4 and 5 queries, 2 should be date-filtered."""
        from workflows.research.web_research.subgraphs.web_researcher import execute_searches

        calls = {"firecrawl": [], "perplexity": []}

        async def mock_firecrawl(query, after_date=None):
            calls["firecrawl"].append({"query": query, "after_date": after_date})
            return []

        async def mock_perplexity(query, after_date=None):
            calls["perplexity"].append({"query": query, "after_date": after_date})
            return []

        state = {
            "search_queries": ["q1", "q2", "q3", "q4", "q5"],
            "recency_filter": {"after_date": "2026-02-01", "quota": 0.4},
        }

        with (
            patch(
                "workflows.research.web_research.subgraphs.web_researcher._search_firecrawl",
                side_effect=mock_firecrawl,
            ),
            patch(
                "workflows.research.web_research.subgraphs.web_researcher._search_perplexity",
                side_effect=mock_perplexity,
            ),
        ):
            await execute_searches(state)

        # quota=0.4 of 5 = round(2.0) = 2 date-filtered queries (indices 0, 1)
        fc_dated = [c for c in calls["firecrawl"] if c["after_date"] is not None]
        fc_undated = [c for c in calls["firecrawl"] if c["after_date"] is None]
        assert len(fc_dated) == 2
        assert len(fc_undated) == 3

    @pytest.mark.asyncio
    async def test_no_recency_filter_means_no_dates(self):
        """Without recency_filter, no calls should have date filtering."""
        from workflows.research.web_research.subgraphs.web_researcher import execute_searches

        calls = []

        async def mock_firecrawl(query, after_date=None):
            calls.append(after_date)
            return []

        async def mock_perplexity(query, after_date=None):
            calls.append(after_date)
            return []

        state = {
            "search_queries": ["q1", "q2"],
            "recency_filter": None,
        }

        with (
            patch(
                "workflows.research.web_research.subgraphs.web_researcher._search_firecrawl",
                side_effect=mock_firecrawl,
            ),
            patch(
                "workflows.research.web_research.subgraphs.web_researcher._search_perplexity",
                side_effect=mock_perplexity,
            ),
        ):
            await execute_searches(state)

        assert all(d is None for d in calls)


# ---------------------------------------------------------------------------
# Web scan query generation
# ---------------------------------------------------------------------------


class TestWebScanQueryGeneration:
    @pytest.mark.asyncio
    async def test_generates_queries_from_llm(self):
        """Should parse JSON array of queries from LLM response."""
        from core.task_queue.workflows.lit_review_web_augmented.phases.web_scan import (
            _generate_search_queries,
        )

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            [
                "latest BCI developments 2026",
                "Neuralink recent clinical trials",
                "brain-computer interface FDA approval 2026",
            ]
        )

        with patch(
            "core.task_queue.workflows.lit_review_web_augmented.phases.web_scan.invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            queries = await _generate_search_queries(
                topic="Brain-Computer Interfaces",
                research_questions=["What progress has been made in BCI clinical trials?"],
            )

        assert len(queries) == 3
        assert "Neuralink" in queries[1]

    @pytest.mark.asyncio
    async def test_fallback_on_bad_json(self):
        """Should fall back to basic queries if LLM returns invalid JSON."""
        from core.task_queue.workflows.lit_review_web_augmented.phases.web_scan import (
            _generate_search_queries,
        )

        mock_response = MagicMock()
        mock_response.content = "Here are some queries:\n1. query one\n2. query two"

        with patch(
            "core.task_queue.workflows.lit_review_web_augmented.phases.web_scan.invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            queries = await _generate_search_queries(
                topic="AI Policy",
                research_questions=["What regulations exist?", "Who are the key players?"],
            )

        # Fallback generates one query per research question
        assert len(queries) == 2
        assert all("latest developments" in q for q in queries)


# ---------------------------------------------------------------------------
# Combine phase
# ---------------------------------------------------------------------------


class TestCombinePhase:
    @pytest.mark.asyncio
    async def test_passes_through_academic_metadata(self):
        """paper_corpus, paper_summaries, zotero_keys must pass through unchanged."""
        from core.task_queue.workflows.lit_review_web_augmented.phases.combine import (
            run_combine_phase,
        )

        lit_result = {
            "final_report": "# Academic Report\n\nSmith et al. (2025) found...",
            "paper_corpus": {"paper1": {"title": "Test Paper"}},
            "paper_summaries": {"paper1": "Summary of test paper"},
            "zotero_keys": ["ZK123"],
        }
        web_result = {
            "final_report": "# Web Research\n\nRecent reports indicate...",
            "source_count": 5,
        }

        mock_response = MagicMock()
        mock_response.content = "# Combined Report\n\nSmith et al. (2025) found... Recent reports indicate..."

        with patch(
            "core.task_queue.workflows.lit_review_web_augmented.phases.combine.invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await run_combine_phase(
                lit_result=lit_result,
                web_result=web_result,
                topic="Test Topic",
                augmented_research_questions=["What is happening?"],
            )

        assert result["paper_corpus"] == {"paper1": {"title": "Test Paper"}}
        assert result["paper_summaries"] == {"paper1": "Summary of test paper"}
        assert result["zotero_keys"] == ["ZK123"]
        assert result["source_breakdown"]["academic"] == 1
        assert result["source_breakdown"]["web"] == 5

    @pytest.mark.asyncio
    async def test_graceful_degradation_no_web(self):
        """If web report is empty, should return academic report unchanged."""
        from core.task_queue.workflows.lit_review_web_augmented.phases.combine import (
            run_combine_phase,
        )

        lit_result = {
            "final_report": "# Academic Report",
            "paper_corpus": {"p1": {}},
            "paper_summaries": {},
            "zotero_keys": [],
        }

        result = await run_combine_phase(
            lit_result=lit_result,
            web_result={"final_report": ""},
            topic="Test",
            augmented_research_questions=["Q1"],
        )

        assert result["final_report"] == "# Academic Report"
        assert result["source_breakdown"]["web"] == 0

    @pytest.mark.asyncio
    async def test_graceful_degradation_no_academic(self):
        """If academic report is empty, should return web report."""
        from core.task_queue.workflows.lit_review_web_augmented.phases.combine import (
            run_combine_phase,
        )

        result = await run_combine_phase(
            lit_result={"final_report": "", "paper_corpus": {}, "paper_summaries": {}, "zotero_keys": []},
            web_result={"final_report": "# Web Report", "source_count": 3},
            topic="Test",
            augmented_research_questions=["Q1"],
        )

        assert result["final_report"] == "# Web Report"
        assert result["source_breakdown"]["academic"] == 0


# ---------------------------------------------------------------------------
# Workflow registration
# ---------------------------------------------------------------------------


class TestWorkflowRegistration:
    def test_workflow_registered(self):
        from core.task_queue.workflows import WORKFLOW_REGISTRY

        assert "lit_review_web_augmented" in WORKFLOW_REGISTRY

    def test_get_workflow(self):
        from core.task_queue.workflows import get_workflow

        workflow = get_workflow("lit_review_web_augmented")
        assert workflow.task_type == "lit_review_web_augmented"

    def test_phases_include_all_expected(self):
        from core.task_queue.workflows import get_phases

        phases = get_phases("lit_review_web_augmented")
        assert "web_scan" in phases
        assert "parallel_research" in phases
        assert "combine" in phases
        assert "supervision" in phases
        assert "evening_reads" in phases
        assert "save_and_spawn" in phases
        assert "complete" in phases

    def test_enum_registered(self):
        from core.task_queue.schemas.enums import TaskType

        assert TaskType.LIT_REVIEW_WEB_AUGMENTED.value == "lit_review_web_augmented"

    def test_topic_task_has_web_scan_window(self):
        """TopicTask TypedDict should include web_scan_window_days."""
        from core.task_queue.schemas.tasks import TopicTask

        assert "web_scan_window_days" in TopicTask.__annotations__


# ---------------------------------------------------------------------------
# Parallel checkpoint resume
# ---------------------------------------------------------------------------


class TestParallelCheckpointResume:
    def test_completed_phases_calculation(self):
        """If resuming from 'combine', web_scan and parallel_research should be skipped."""
        from core.task_queue.workflows.lit_review_web_augmented import LitReviewWebAugmentedWorkflow

        workflow = LitReviewWebAugmentedWorkflow()
        checkpoint = {"phase": "combine"}
        completed = workflow._get_completed_phases(checkpoint)

        assert "web_scan" in completed
        assert "parallel_research" in completed
        assert "combine" not in completed  # current phase not in completed set

    def test_resume_from_parallel_research(self):
        """If resuming from parallel_research, only web_scan should be skipped."""
        from core.task_queue.workflows.lit_review_web_augmented import LitReviewWebAugmentedWorkflow

        workflow = LitReviewWebAugmentedWorkflow()
        checkpoint = {"phase": "parallel_research"}
        completed = workflow._get_completed_phases(checkpoint)

        assert "web_scan" in completed
        assert "parallel_research" not in completed


# ---------------------------------------------------------------------------
# RecencyFilter type
# ---------------------------------------------------------------------------


class TestRecencyFilterType:
    def test_recency_filter_fields(self):
        from workflows.research.web_research.state import RecencyFilter

        rf = RecencyFilter(after_date="2026-02-01", quota=0.3)
        assert rf["after_date"] == "2026-02-01"
        assert rf["quota"] == 0.3
