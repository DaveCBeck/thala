"""Tests for fallback queue threading through the academic lit review workflow.

Verifies:
1. FallbackManager works when corpus includes fallback DOIs
2. filter_by_relevance_node includes overflow/near-threshold in discovered_papers
   while keyword_dois remains selected-only
"""

from unittest.mock import AsyncMock, patch

import pytest

from workflows.research.academic_lit_review.paper_processor.fallback import (
    FallbackManager,
)
from workflows.research.academic_lit_review.state import (
    FallbackCandidate,
    PaperMetadata,
)


def _make_paper(doi: str, title: str = "Test", score: float = 0.8) -> PaperMetadata:
    """Create a minimal PaperMetadata for testing."""
    return PaperMetadata(
        doi=doi,
        title=title,
        authors=[],
        publication_date=None,
        year=2024,
        venue=None,
        cited_by_count=0,
        abstract="Abstract",
        openalex_id=f"https://openalex.org/{doi}",
        primary_topic=None,
        is_oa=True,
        oa_url=f"https://example.com/{doi}.pdf",
        oa_urls=[f"https://example.com/{doi}.pdf"],
        pmcid=None,
        oa_status="gold",
        referenced_works=[],
        citing_works_count=0,
        retrieved_at=None,
        discovery_stage=0,
        discovery_method="keyword",
        relevance_score=score,
    )


class TestFallbackManagerWithCorpus:
    """FallbackManager returns metadata when fallback DOIs are in the corpus."""

    def test_get_fallback_returns_metadata_from_corpus(self):
        fallback_paper = _make_paper("10.1234/fallback1", "Fallback Paper", 0.65)
        corpus = {
            "10.1234/primary1": _make_paper("10.1234/primary1"),
            "10.1234/fallback1": fallback_paper,
        }
        queue = [
            FallbackCandidate(doi="10.1234/fallback1", relevance_score=0.65, source="overflow"),
        ]

        manager = FallbackManager(fallback_queue=queue, paper_corpus=corpus)
        result = manager.get_fallback_for(
            failed_doi="10.1234/primary1",
            failure_reason="acquisition_failed",
            failure_stage="acquisition",
        )

        assert result is not None
        assert result["doi"] == "10.1234/fallback1"
        assert result["title"] == "Fallback Paper"

    def test_get_fallback_returns_none_when_queue_empty(self):
        corpus = {"10.1234/primary1": _make_paper("10.1234/primary1")}
        manager = FallbackManager(fallback_queue=[], paper_corpus=corpus)

        result = manager.get_fallback_for(
            failed_doi="10.1234/primary1",
            failure_reason="acquisition_failed",
            failure_stage="acquisition",
        )

        assert result is None
        assert "10.1234/primary1" in manager.get_exhausted_warnings()

    def test_get_fallback_skips_candidate_missing_from_corpus(self):
        """If a candidate DOI has no metadata in corpus, skip to next."""
        good_paper = _make_paper("10.1234/fallback2", "Good Fallback", 0.55)
        corpus = {
            # fallback1 intentionally missing from corpus
            "10.1234/fallback2": good_paper,
        }
        queue = [
            FallbackCandidate(doi="10.1234/fallback1", relevance_score=0.65, source="overflow"),
            FallbackCandidate(doi="10.1234/fallback2", relevance_score=0.55, source="near_threshold"),
        ]

        manager = FallbackManager(fallback_queue=queue, paper_corpus=corpus)
        result = manager.get_fallback_for(
            failed_doi="10.1234/primary1",
            failure_reason="pdf_invalid",
            failure_stage="marker",
        )

        assert result is not None
        assert result["doi"] == "10.1234/fallback2"

    def test_substitutions_tracked(self):
        corpus = {"10.1234/fb": _make_paper("10.1234/fb", score=0.62)}
        queue = [FallbackCandidate(doi="10.1234/fb", relevance_score=0.62, source="overflow")]

        manager = FallbackManager(fallback_queue=queue, paper_corpus=corpus)
        manager.get_fallback_for("10.1234/failed", "acquisition_failed", "acquisition")

        subs = manager.get_substitutions()
        assert len(subs) == 1
        assert subs[0]["failed_doi"] == "10.1234/failed"
        assert subs[0]["fallback_doi"] == "10.1234/fb"

    def test_merged_corpus_prefers_existing_entries(self):
        """Simulates nodes.py merge: paper_corpus.setdefault doesn't overwrite."""
        corpus = {"10.1234/shared": _make_paper("10.1234/shared", "From Corpus", 0.7)}
        process_paper = _make_paper("10.1234/shared", "From Process", 0.7)

        # Simulate the merge logic from nodes.py
        merged = dict(corpus)
        merged.setdefault(process_paper["doi"], process_paper)

        assert merged["10.1234/shared"]["title"] == "From Corpus"


class TestFilterByRelevanceNode:
    """filter_by_relevance_node includes overflow/near-threshold in discovered_papers."""

    @pytest.mark.asyncio
    async def test_discovered_papers_includes_all_keyword_dois_selected_only(self):
        """discovered_papers has selected + overflow + near-threshold;
        keyword_dois has only selected."""
        from workflows.research.academic_lit_review.keyword_search.searcher import (
            filter_by_relevance_node,
        )

        selected = [_make_paper(f"10.1234/sel{i}", score=0.9) for i in range(3)]
        overflow = [_make_paper(f"10.1234/over{i}", score=0.7) for i in range(2)]
        near_threshold = [_make_paper(f"10.1234/near{i}", score=0.55) for i in range(2)]

        state = {
            "raw_results": ["dummy"],  # Non-empty to pass early return
            "input": {"topic": "test topic", "research_questions": []},
            "quality_settings": {"max_papers": 3},
            "language_config": None,
        }

        with (
            patch(
                "workflows.research.academic_lit_review.keyword_search.searcher.convert_to_paper_metadata",
                side_effect=lambda **kw: _make_paper("10.1234/dummy"),
            ),
            patch(
                "workflows.research.academic_lit_review.keyword_search.searcher.deduplicate_papers",
                return_value=[_make_paper("10.1234/dummy")],
            ),
            patch(
                "workflows.research.academic_lit_review.keyword_search.searcher.batch_score_relevance",
                new_callable=AsyncMock,
                return_value=(
                    selected + overflow,  # relevant (5 papers, exceeds max_papers=3)
                    near_threshold,  # fallback_candidates
                    [],  # rejected
                ),
            ),
        ):
            result = await filter_by_relevance_node(state)

        # keyword_dois: only the 3 selected papers
        assert len(result["keyword_dois"]) == 3
        for doi in result["keyword_dois"]:
            assert doi.startswith("10.1234/sel")

        # discovered_papers: selected + overflow + near-threshold = 3 + 2 + 2 = 7
        assert len(result["discovered_papers"]) == 7

        # fallback_queue: overflow (2) + near-threshold (2) = 4
        assert len(result["fallback_queue"]) == 4
        overflow_entries = [f for f in result["fallback_queue"] if f["source"] == "overflow"]
        near_entries = [f for f in result["fallback_queue"] if f["source"] == "near_threshold"]
        assert len(overflow_entries) == 2
        assert len(near_entries) == 2
