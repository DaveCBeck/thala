"""Tests for overgeneration: generate_candidate, select_per_location, routing, finalize."""

from unittest.mock import AsyncMock, patch

import pytest

from workflows.output.illustrate.config import IllustrateConfig
from workflows.output.illustrate.graph import (
    _FALLBACK_IMAGE_TYPE,
    route_after_analysis,
    route_after_selection,
    route_to_selection,
)
from workflows.output.illustrate.nodes.finalize import _select_winning_results
from workflows.output.illustrate.nodes.generate_candidate import generate_candidate_node
from workflows.output.illustrate.nodes.select_per_location import (
    select_per_location_node,
)
from workflows.output.illustrate.schemas import (
    CandidateBrief,
    ImageLocationPlan,
    VisualIdentity,
)
from workflows.output.illustrate.state import ImageGenResult, LocationSelection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(**overrides):
    defaults = dict(
        location_id="section_1",
        insertion_after_header="Introduction",
        purpose="illustration",
        image_type="generated",
        brief="A striking image",
    )
    defaults.update(overrides)
    return ImageLocationPlan(**defaults)


def _make_brief(**overrides):
    defaults = dict(
        location_id="section_1",
        candidate_index=1,
        image_type="generated",
        brief="A striking image",
        relationship_to_text="evocative",
        visual_identity_references="warm palette",
    )
    defaults.update(overrides)
    return CandidateBrief(**defaults)


def _make_gen_result(**overrides):
    defaults = dict(
        location_id="section_1",
        brief_id="section_1_1",
        success=True,
        image_bytes=b"PNG_DATA",
        image_type="generated",
        prompt_or_query_used="test prompt",
        alt_text="Test image",
        attribution=None,
    )
    defaults.update(overrides)
    return ImageGenResult(**defaults)


def _make_vi():
    return VisualIdentity(
        primary_style="editorial watercolor",
        color_palette=["warm amber", "deep teal"],
        mood="contemplative",
        lighting="soft diffused",
        avoid=["neon colors"],
    )


# ---------------------------------------------------------------------------
# generate_candidate_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGenerateCandidateNode:
    async def test_routes_to_public_domain(self):
        state = {
            "location": _make_plan(image_type="public_domain"),
            "brief": _make_brief(image_type="public_domain"),
            "brief_id": "section_1_1",
            "document_context": "test content",
            "config": IllustrateConfig(),
        }

        mock_result = {
            "generation_results": [
                _make_gen_result(image_type="public_domain", brief_id=""),
            ]
        }

        with patch(
            "workflows.output.illustrate.nodes.generate_candidate._generate_public_domain",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await generate_candidate_node(state)

        assert result["generation_results"][0]["brief_id"] == "section_1_1"
        assert result["generation_results"][0]["image_type"] == "public_domain"

    async def test_routes_to_diagram(self):
        state = {
            "location": _make_plan(image_type="diagram", diagram_subtype="flowchart"),
            "brief": _make_brief(image_type="diagram", diagram_subtype="flowchart"),
            "brief_id": "section_1_1",
            "document_context": "test content",
            "config": IllustrateConfig(),
        }

        mock_result = {
            "generation_results": [
                _make_gen_result(image_type="diagram", brief_id=""),
            ]
        }

        with patch(
            "workflows.output.illustrate.nodes.generate_candidate._generate_diagram",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await generate_candidate_node(state)

        assert result["generation_results"][0]["brief_id"] == "section_1_1"

    async def test_routes_to_imagen(self):
        state = {
            "location": _make_plan(image_type="generated"),
            "brief": _make_brief(image_type="generated"),
            "brief_id": "section_1_1",
            "document_context": "test content",
            "config": IllustrateConfig(),
            "visual_identity": _make_vi(),
        }

        mock_result = {
            "generation_results": [
                _make_gen_result(image_type="generated", brief_id=""),
            ]
        }

        with patch(
            "workflows.output.illustrate.nodes.generate_candidate._generate_imagen",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await generate_candidate_node(state)

        assert result["generation_results"][0]["brief_id"] == "section_1_1"

    async def test_exception_returns_failure(self):
        state = {
            "location": _make_plan(),
            "brief": _make_brief(),
            "brief_id": "section_1_1",
            "document_context": "test content",
            "config": IllustrateConfig(),
        }

        with patch(
            "workflows.output.illustrate.nodes.generate_candidate._generate_imagen",
            new_callable=AsyncMock,
            side_effect=Exception("API timeout"),
        ):
            result = await generate_candidate_node(state)

        gen = result["generation_results"][0]
        assert gen["success"] is False
        assert gen["brief_id"] == "section_1_1"
        assert len(result["errors"]) == 1


# ---------------------------------------------------------------------------
# select_per_location_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSelectPerLocationNode:
    async def test_zero_candidates_returns_failed(self):
        state = {"location_id": "s1", "candidates": []}
        result = await select_per_location_node(state)

        sel = result["selection_results"][0]
        assert sel["quality_tier"] == "failed"
        assert sel["selected_brief_id"] is None
        assert sel["location_id"] == "s1"

    async def test_one_candidate_auto_selects(self):
        candidate = _make_gen_result(brief_id="s1_1")
        state = {"location_id": "s1", "candidates": [candidate]}
        result = await select_per_location_node(state)

        sel = result["selection_results"][0]
        assert sel["quality_tier"] == "acceptable"
        assert sel["selected_brief_id"] == "s1_1"

    async def test_two_candidates_uses_vision(self):
        c1 = _make_gen_result(brief_id="s1_1", image_bytes=b"IMG_A")
        c2 = _make_gen_result(brief_id="s1_2", image_bytes=b"IMG_B")
        state = {
            "location_id": "s1",
            "candidates": [c1, c2],
            "selection_criteria": "Pick the better one",
        }

        with patch(
            "workflows.output.illustrate.nodes.select_per_location.vision_pair_select",
            new_callable=AsyncMock,
            return_value=1,  # Second candidate wins
        ):
            result = await select_per_location_node(state)

        sel = result["selection_results"][0]
        assert sel["quality_tier"] == "excellent"
        assert sel["selected_brief_id"] == "s1_2"

    async def test_vision_failure_falls_back_to_first(self):
        c1 = _make_gen_result(brief_id="s1_1", image_bytes=b"IMG_A")
        c2 = _make_gen_result(brief_id="s1_2", image_bytes=b"IMG_B")
        state = {"location_id": "s1", "candidates": [c1, c2]}

        with patch(
            "workflows.output.illustrate.nodes.select_per_location.vision_pair_select",
            new_callable=AsyncMock,
            side_effect=Exception("Vision API down"),
        ):
            result = await select_per_location_node(state)

        sel = result["selection_results"][0]
        assert sel["quality_tier"] == "acceptable"
        assert sel["selected_brief_id"] == "s1_1"


# ---------------------------------------------------------------------------
# Routing: route_after_analysis
# ---------------------------------------------------------------------------


class TestRouteAfterAnalysis:
    def test_failed_status_goes_to_finalize(self):
        state = {"status": "failed", "input": {"markdown_document": ""}}
        assert route_after_analysis(state) == "finalize"

    def test_no_briefs_goes_to_finalize(self):
        state = {
            "candidate_briefs": [],
            "image_plan": [],
            "input": {"markdown_document": ""},
        }
        assert route_after_analysis(state) == "finalize"

    def test_fans_out_per_brief(self):
        brief1 = _make_brief(location_id="header", candidate_index=1)
        brief2 = _make_brief(location_id="header", candidate_index=2, image_type="public_domain")
        brief3 = _make_brief(location_id="s1", candidate_index=1, image_type="diagram")
        plan_h = _make_plan(location_id="header", purpose="header")
        plan_s1 = _make_plan(location_id="s1")

        state = {
            "candidate_briefs": [brief1, brief2, brief3],
            "image_plan": [plan_h, plan_s1],
            "input": {"markdown_document": "test doc"},
        }

        result = route_after_analysis(state)
        assert len(result) == 3
        # All go to generate_candidate
        for send in result:
            assert send.node == "generate_candidate"

        # Check brief_id tagging
        brief_ids = [s.arg["brief_id"] for s in result]
        assert "header_1" in brief_ids
        assert "header_2" in brief_ids
        assert "s1_1" in brief_ids

    def test_skips_briefs_without_plan(self):
        brief = _make_brief(location_id="orphan")
        plan = _make_plan(location_id="s1")

        state = {
            "candidate_briefs": [brief],
            "image_plan": [plan],
            "input": {"markdown_document": "test"},
        }

        result = route_after_analysis(state)
        # No sends because brief has no matching plan
        assert result == "finalize"


# ---------------------------------------------------------------------------
# Routing: route_to_selection
# ---------------------------------------------------------------------------


class TestRouteToSelection:
    def test_no_results_goes_to_finalize(self):
        state = {"generation_results": [], "candidate_briefs": []}
        assert route_to_selection(state) == "finalize"

    def test_groups_by_location_and_filters_successful(self):
        results = [
            _make_gen_result(location_id="s1", brief_id="s1_1", success=True),
            _make_gen_result(location_id="s1", brief_id="s1_2", success=False, image_bytes=None),
            _make_gen_result(location_id="s2", brief_id="s2_1", success=True),
        ]
        briefs = [_make_brief(location_id="s1"), _make_brief(location_id="s2")]

        state = {"generation_results": results, "candidate_briefs": briefs}
        sends = route_to_selection(state)

        assert len(sends) == 2
        for send in sends:
            assert send.node == "select_per_location"

        # s1 should have 1 successful candidate (s1_2 failed)
        s1_send = next(s for s in sends if s.arg["location_id"] == "s1")
        assert len(s1_send.arg["candidates"]) == 1


# ---------------------------------------------------------------------------
# Routing: route_after_selection
# ---------------------------------------------------------------------------


class TestRouteAfterSelection:
    def test_no_failures_goes_to_finalize(self):
        state = {
            "selection_results": [
                LocationSelection(
                    location_id="s1",
                    selected_brief_id="s1_1",
                    quality_tier="excellent",
                    reasoning="good",
                ),
            ],
            "retry_count": {},
            "image_plan": [],
            "candidate_briefs": [],
            "input": {"markdown_document": ""},
        }
        assert route_after_selection(state) == "finalize"

    def test_retry_with_cross_strategy_fallback(self):
        briefs = [
            _make_brief(location_id="s1", candidate_index=1, image_type="public_domain"),
            _make_brief(location_id="s1", candidate_index=2, image_type="public_domain"),
        ]
        state = {
            "selection_results": [
                LocationSelection(
                    location_id="s1",
                    selected_brief_id=None,
                    quality_tier="failed",
                    reasoning="both failed",
                ),
            ],
            "retry_count": {"s1": 1},
            "config": IllustrateConfig(max_retries=1),
            "image_plan": [_make_plan(location_id="s1")],
            "candidate_briefs": briefs,
            "input": {"markdown_document": "test"},
        }

        sends = route_after_selection(state)
        assert len(sends) == 2
        # Both should use fallback type (generated, not public_domain)
        for send in sends:
            assert send.arg["brief"].image_type == "generated"

    def test_exceeds_retry_limit_goes_to_finalize(self):
        state = {
            "selection_results": [
                LocationSelection(
                    location_id="s1",
                    selected_brief_id=None,
                    quality_tier="failed",
                    reasoning="both failed",
                ),
            ],
            "retry_count": {"s1": 2},
            "config": IllustrateConfig(max_retries=1),
            "image_plan": [_make_plan(location_id="s1")],
            "candidate_briefs": [_make_brief(location_id="s1")],
            "input": {"markdown_document": ""},
        }
        assert route_after_selection(state) == "finalize"


# ---------------------------------------------------------------------------
# Cross-strategy fallback map
# ---------------------------------------------------------------------------


class TestFallbackMap:
    def test_pd_falls_back_to_generated(self):
        assert _FALLBACK_IMAGE_TYPE["public_domain"] == "generated"

    def test_generated_falls_back_to_pd(self):
        assert _FALLBACK_IMAGE_TYPE["generated"] == "public_domain"

    def test_diagram_falls_back_to_generated(self):
        assert _FALLBACK_IMAGE_TYPE["diagram"] == "generated"


# ---------------------------------------------------------------------------
# Finalize: _select_winning_results
# ---------------------------------------------------------------------------


class TestSelectWinningResults:
    def test_picks_selected_brief_id(self):
        gen_results = [
            _make_gen_result(location_id="s1", brief_id="s1_1", image_bytes=b"A"),
            _make_gen_result(location_id="s1", brief_id="s1_2", image_bytes=b"B"),
        ]
        selection = [
            LocationSelection(
                location_id="s1",
                selected_brief_id="s1_2",
                quality_tier="excellent",
                reasoning="better",
            ),
        ]

        winners = _select_winning_results(gen_results, selection)
        assert len(winners) == 1
        assert winners[0]["brief_id"] == "s1_2"

    def test_failed_selection_produces_no_winner(self):
        gen_results = [
            _make_gen_result(location_id="s1", brief_id="s1_1", success=False, image_bytes=None),
        ]
        selection = [
            LocationSelection(
                location_id="s1",
                selected_brief_id=None,
                quality_tier="failed",
                reasoning="both failed",
            ),
        ]

        winners = _select_winning_results(gen_results, selection)
        assert len(winners) == 0

    def test_locations_without_selection_included(self):
        """Results from retry rounds (without selection) should still be included."""
        gen_results = [
            _make_gen_result(location_id="retry_loc", brief_id="retry_loc_1_retry"),
        ]
        selection = []  # No selection for this location

        winners = _select_winning_results(gen_results, selection)
        assert len(winners) == 1
        assert winners[0]["location_id"] == "retry_loc"

    def test_deduplicates_selection_per_location(self):
        gen_results = [
            _make_gen_result(location_id="s1", brief_id="s1_1"),
        ]
        # Duplicate selection entries for same location
        selection = [
            LocationSelection(
                location_id="s1",
                selected_brief_id="s1_1",
                quality_tier="acceptable",
                reasoning="auto",
            ),
            LocationSelection(
                location_id="s1",
                selected_brief_id="s1_1",
                quality_tier="acceptable",
                reasoning="duplicate",
            ),
        ]

        winners = _select_winning_results(gen_results, selection)
        assert len(winners) == 1
