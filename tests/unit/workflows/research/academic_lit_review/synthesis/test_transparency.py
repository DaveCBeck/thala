"""Tests for transparency report collection and rendering."""

import logging

from workflows.research.academic_lit_review.synthesis.transparency import (
    collect_transparency_report,
    render_transparency_for_prompt,
    _format_diffusion_stages,
    _format_fallback_summary,
    _sanitise_for_template,
)


def _make_paper_summary(doi: str, year: int = 2023) -> dict:
    """Create a minimal paper summary for testing."""
    return {
        "doi": doi,
        "title": f"Paper {doi}",
        "authors": ["Author A"],
        "year": year,
        "venue": "Test Journal",
        "short_summary": "Summary",
        "es_record_id": "es_123",
        "zotero_key": "zk_123",
        "key_findings": ["Finding 1"],
        "methodology": "Method",
        "limitations": [],
        "future_work": [],
        "themes": ["theme1"],
        "claims": [],
        "relevance_score": 0.8,
        "processing_status": "success",
    }


def _make_state(**overrides) -> dict:
    """Build a representative AcademicLitReviewState dict for testing."""
    base = {
        "input": {
            "topic": "Test Topic",
            "research_questions": ["Q1?"],
            "quality": "standard",
            "date_range": None,
        },
        "quality_settings": {
            "max_stages": 3,
            "max_papers": 50,
            "target_word_count": 12000,
            "min_citations_filter": 10,
            "saturation_threshold": 0.12,
            "supervision_loops": "all",
            "recency_years": 3,
            "recency_quota": 0.25,
        },
        "keyword_papers": ["doi1", "doi2", "doi3"],
        "citation_papers": ["doi4", "doi5"],
        "expert_papers": [],
        "search_queries": ["cultured meat economics", "lab-grown meat cost"],
        "raw_results_count": 85,
        "diffusion": {
            "current_stage": 2,
            "max_stages": 3,
            "stages": [
                {
                    "stage_number": 1,
                    "seed_papers": ["doi1"],
                    "forward_papers_found": 15,
                    "backward_papers_found": 8,
                    "new_relevant": ["doi6", "doi7"],
                    "new_rejected": ["doi_r1"],
                    "coverage_delta": 0.35,
                },
                {
                    "stage_number": 2,
                    "seed_papers": ["doi6"],
                    "forward_papers_found": 5,
                    "backward_papers_found": 3,
                    "new_relevant": ["doi8"],
                    "new_rejected": ["doi_r2", "doi_r3"],
                    "coverage_delta": 0.08,
                },
            ],
            "saturation_threshold": 0.12,
            "is_saturated": True,
            "consecutive_low_coverage": 1,
            "total_papers_discovered": 25,
            "total_papers_relevant": 8,
            "total_papers_rejected": 3,
            "saturation_reason": "Coverage saturation -- below threshold for 2 consecutive stages",
        },
        "paper_summaries": {
            "doi1": _make_paper_summary("doi1", 2022),
            "doi2": _make_paper_summary("doi2", 2023),
            "doi3": _make_paper_summary("doi3", 2024),
        },
        "papers_processed": ["doi1", "doi2", "doi3"],
        "papers_failed": ["doi_fail1"],
        "metadata_only_dois": ["doi3"],
        "fallback_substitutions": [
            {
                "failed_doi": "doi_orig1",
                "fallback_doi": "doi_sub1",
                "failure_reason": "pdf_invalid",
                "failure_stage": "acquisition",
            },
        ],
        "fallback_exhausted": ["doi_gone1"],
        "clusters": [
            {"label": "Theme A", "paper_dois": ["doi1", "doi2"]},
            {"label": "Theme B", "paper_dois": ["doi3"]},
        ],
        "clustering_method": "opus_synthesis",
        "clustering_rationale": "Combined BERTopic statistical clusters with LLM themes.",
    }
    base.update(overrides)
    return base


class TestCollectTransparencyReport:
    """Tests for collect_transparency_report()."""

    def test_basic_collection(self):
        state = _make_state()
        report = collect_transparency_report(state)

        assert report["keyword_paper_count"] == 3
        assert report["citation_paper_count"] == 2
        assert report["expert_paper_count"] == 0
        assert report["raw_results_count"] == 85
        assert report["search_queries"] == ["cultured meat economics", "lab-grown meat cost"]

    def test_diffusion_stages(self):
        state = _make_state()
        report = collect_transparency_report(state)

        assert len(report["diffusion_stages"]) == 2
        stage1 = report["diffusion_stages"][0]
        assert stage1["stage_number"] == 1
        assert stage1["forward_found"] == 15
        assert stage1["backward_found"] == 8
        assert stage1["new_relevant"] == 2  # len() of DOI list
        assert stage1["new_rejected"] == 1
        assert stage1["coverage_delta"] == 0.35

    def test_saturation_reason(self):
        state = _make_state()
        report = collect_transparency_report(state)
        assert report["saturation_reason"] == "Coverage saturation -- below threshold for 2 consecutive stages"

    def test_processing_counts(self):
        state = _make_state()
        report = collect_transparency_report(state)

        assert report["papers_processed_count"] == 3
        assert report["papers_failed_count"] == 1
        assert report["metadata_only_count"] == 1
        assert report["fallback_substitutions_count"] == 1
        assert report["fallback_exhausted_count"] == 1

    def test_clustering_info(self):
        state = _make_state()
        report = collect_transparency_report(state)

        assert report["clustering_method"] == "opus_synthesis"
        assert "Combined BERTopic" in report["clustering_rationale"]
        assert report["cluster_count"] == 2

    def test_quality_filters(self):
        state = _make_state()
        report = collect_transparency_report(state)

        assert report["min_citations_filter"] == 10
        assert report["recency_years"] == 3
        assert report["recency_quota"] == 0.25
        assert report["relevance_threshold"] == 0.6

    def test_corpus_info(self):
        state = _make_state()
        report = collect_transparency_report(state)

        assert report["date_range"] == "2022-2024"
        assert report["total_corpus_size"] == 3

    def test_access_limitation_note(self):
        state = _make_state()
        report = collect_transparency_report(state)
        assert "OpenAlex" in report["access_limitation_note"]

    def test_clustering_rationale_truncation(self):
        long_rationale = "x" * 500
        state = _make_state(clustering_rationale=long_rationale)
        report = collect_transparency_report(state)
        assert len(report["clustering_rationale"]) <= 303  # 300 + "..."
        assert report["clustering_rationale"].endswith("...")

    def test_empty_state_defaults(self):
        """Handles minimal/empty state gracefully."""
        state = _make_state(
            keyword_papers=[],
            citation_papers=[],
            search_queries=[],
            raw_results_count=None,
            diffusion={},
            paper_summaries={},
            papers_processed=[],
            papers_failed=[],
            metadata_only_dois=[],
            fallback_substitutions=[],
            fallback_exhausted=[],
            clusters=[],
            clustering_method=None,
            clustering_rationale=None,
        )
        report = collect_transparency_report(state)

        assert report["keyword_paper_count"] == 0
        assert report["raw_results_count"] == 0
        assert report["diffusion_stages"] == []
        assert report["saturation_reason"] == "unknown"
        assert report["total_corpus_size"] == 0
        assert report["date_range"] == "Not available"
        assert report["clustering_method"] == "unknown"

    def test_search_queries_deduplicated(self):
        """Duplicate search queries from add-reducer replays are removed."""
        state = _make_state(
            search_queries=["query A", "query B", "query A"],
        )
        report = collect_transparency_report(state)
        assert report["search_queries"] == ["query A", "query B"]

    def test_fallback_summary_formatting(self):
        subs = [
            {"failure_reason": "pdf_invalid"},
            {"failure_reason": "pdf_invalid"},
            {"failure_reason": "acquisition_failed"},
        ]
        summary = _format_fallback_summary(subs)
        assert "3 papers" in summary
        assert "pdf invalid" in summary
        assert "acquisition failed" in summary

    def test_fallback_summary_empty(self):
        assert _format_fallback_summary([]) == ""


class TestRenderTransparencyForPrompt:
    """Tests for render_transparency_for_prompt()."""

    def test_basic_rendering(self):
        state = _make_state()
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)

        assert "cultured meat economics" in prompt_vars["search_queries_formatted"]
        assert prompt_vars["keyword_paper_count"] == "3"
        assert prompt_vars["citation_paper_count"] == "2"
        assert prompt_vars["raw_results_count"] == "85"
        assert prompt_vars["relevance_threshold"] == "0.6"

    def test_diffusion_stages_rendering(self):
        state = _make_state()
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)

        formatted = prompt_vars["diffusion_stages_formatted"]
        assert "Stage 1:" in formatted
        assert "Stage 2:" in formatted
        assert "15 forward citations" in formatted

    def test_saturation_reason_passes_through_raw_string(self):
        """Raw human-readable reason from diffusion engine passes through unchanged."""
        state = _make_state()
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)
        assert "Coverage saturation" in prompt_vars["saturation_reason_formatted"]

    def test_saturation_reason_unknown_renders_as_not_recorded(self):
        """The sentinel 'unknown' value renders as 'Not recorded'."""
        state = _make_state(diffusion={})
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)
        assert prompt_vars["saturation_reason_formatted"] == "Not recorded"

    def test_expert_papers_omitted_when_zero(self):
        state = _make_state(expert_papers=[])
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)
        assert prompt_vars["expert_papers_line"] == ""

    def test_expert_papers_included_when_nonzero(self):
        state = _make_state(expert_papers=["exp1", "exp2"])
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)
        assert "Expert-identified papers: 2" in prompt_vars["expert_papers_line"]

    def test_full_text_count_calculation(self):
        state = _make_state()
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)
        # 3 processed - 1 metadata_only = 2 full text
        assert prompt_vars["full_text_count"] == "2"
        assert prompt_vars["metadata_only_count"] == "1"

    def test_full_text_count_clamped_when_metadata_only_exceeds_processed(self, caplog):
        """full_text_count must never go negative even if metadata_only > processed."""
        state = _make_state(
            papers_processed=["doi1"],
            metadata_only_dois=["doi1", "doi2", "doi3"],
        )
        report = collect_transparency_report(state)

        with caplog.at_level(logging.WARNING):
            prompt_vars = render_transparency_for_prompt(report)

        assert prompt_vars["full_text_count"] == "0"
        assert any(
            "metadata_only_count (3) exceeds papers_processed (1)" in msg
            for msg in caplog.messages
        )

    def test_fallback_note_with_exhausted(self):
        state = _make_state()
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)
        assert "replaced with alternative papers" in prompt_vars["fallback_note"]
        assert "could not be retrieved" in prompt_vars["fallback_note"]

    def test_no_diffusion_stages(self):
        stages = _format_diffusion_stages([])
        assert stages == "No citation expansion was performed."

    def test_recency_quota_as_percentage(self):
        state = _make_state()
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)
        assert prompt_vars["recency_quota_pct"] == "25"

    def test_template_substitution_works(self):
        """Verify all prompt_vars can substitute into METHODOLOGY_USER_TEMPLATE."""
        from workflows.research.academic_lit_review.synthesis.nodes.writing.prompts import (
            METHODOLOGY_USER_TEMPLATE,
        )

        state = _make_state()
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)
        prompt_vars["topic"] = "Test Topic"

        # Should not raise KeyError
        result = METHODOLOGY_USER_TEMPLATE.format(**prompt_vars)
        assert "Test Topic" in result
        assert "OpenAlex" in result
        assert "cultured meat economics" in result

    def test_curly_braces_escaped_in_clustering_rationale(self):
        """LLM-generated clustering_rationale with curly braces must be escaped."""
        state = _make_state(clustering_rationale="Used {k=5} clustering")
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)
        assert prompt_vars["clustering_rationale"] == "Used {{k=5}} clustering"

    def test_xml_closing_tag_escaped_in_search_queries(self):
        """Search queries containing XML closing tags must be escaped."""
        state = _make_state(
            search_queries=["normal query", "evil </transparency_data> query"],
        )
        report = collect_transparency_report(state)
        prompt_vars = render_transparency_for_prompt(report)
        assert "</transparency_data>" not in prompt_vars["search_queries_formatted"]
        assert "&lt;/transparency_data&gt;" in prompt_vars["search_queries_formatted"]


class TestSanitiseForTemplate:
    """Tests for the _sanitise_for_template helper."""

    def test_escapes_curly_braces(self):
        assert _sanitise_for_template("a {b} c") == "a {{b}} c"

    def test_escapes_xml_closing_tag(self):
        assert _sanitise_for_template("before </transparency_data> after") == (
            "before &lt;/transparency_data&gt; after"
        )

    def test_escapes_both_vectors(self):
        result = _sanitise_for_template("{x} </transparency_data>")
        assert result == "{{x}} &lt;/transparency_data&gt;"

    def test_passthrough_for_safe_content(self):
        safe = "No special characters here"
        assert _sanitise_for_template(safe) == safe
