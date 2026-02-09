"""Tests for editorial review: assemble_document, editorial_review, finalize filtering."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflows.output.illustrate.config import IllustrateConfig
from workflows.output.illustrate.graph import route_after_assembly, route_after_selection
from workflows.output.illustrate.nodes.assemble_document import (
    assemble_document_node,
)
from workflows.output.illustrate.nodes.editorial_review import (
    _compute_cuts_count,
    editorial_review_node,
)
from workflows.output.illustrate.nodes.finalize import _determine_status, finalize_node
from workflows.output.illustrate.schemas import (
    EditorialImageEvaluation,
    EditorialReviewResult,
)
from workflows.output.illustrate.state import FinalImage, LocationSelection

from .conftest import (
    _make_assembled_image,
    _make_gen_result,
    _make_opportunity,
    _make_plan,
    _make_vi,
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestEditorialReviewSchemas:
    def test_editorial_image_evaluation_valid(self):
        ev = EditorialImageEvaluation(
            location_id="section_1",
            contribution_rank=1,
            visual_coherence=4,
            pacing_contribution=3,
            variety_contribution=5,
            individual_quality=4,
        )
        assert ev.location_id == "section_1"
        assert ev.cut_reason is None

    def test_editorial_image_evaluation_with_cut_reason(self):
        ev = EditorialImageEvaluation(
            location_id="section_3",
            contribution_rank=6,
            visual_coherence=2,
            pacing_contribution=2,
            variety_contribution=1,
            individual_quality=3,
            cut_reason="Too similar to adjacent image",
        )
        assert ev.cut_reason == "Too similar to adjacent image"

    def test_editorial_review_result_valid(self):
        result = EditorialReviewResult(
            evaluations=[
                EditorialImageEvaluation(
                    location_id="s1",
                    contribution_rank=1,
                    visual_coherence=5,
                    pacing_contribution=5,
                    variety_contribution=5,
                    individual_quality=5,
                ),
            ],
            cut_location_ids=["s3"],
            editorial_summary="Good set overall",
        )
        assert result.cut_location_ids == ["s3"]

    def test_editorial_review_result_json_string_list(self):
        """LLM may return JSON string instead of list."""
        result = EditorialReviewResult(
            evaluations="[]",
            cut_location_ids='["s3", "s5"]',
            editorial_summary="summary",
        )
        assert result.cut_location_ids == ["s3", "s5"]
        assert result.evaluations == []

    def test_editorial_review_result_score_bounds(self):
        """Scores must be 1-5."""
        with pytest.raises(Exception):
            EditorialImageEvaluation(
                location_id="s1",
                contribution_rank=1,
                visual_coherence=0,  # Below min
                pacing_contribution=3,
                variety_contribution=3,
                individual_quality=3,
            )


# ---------------------------------------------------------------------------
# _compute_cuts_count
# ---------------------------------------------------------------------------


class TestComputeCutsCount:
    def test_standard_surplus(self):
        """8 opportunities (6 non-header + 2 surplus), 6 surviving → cut 2."""
        non_header = [_make_assembled_image(location_id=f"s{i}") for i in range(6)]
        opportunities = [
            _make_opportunity(location_id="header", purpose="header"),
        ] + [_make_opportunity(location_id=f"s{i}") for i in range(6)]
        # Target = 6 non-header opps - 2 = 4
        assert _compute_cuts_count(non_header, opportunities) == 2

    def test_exact_target_no_cuts(self):
        """4 non-header opps, 2 surviving → target=2, surplus=0 → cut 0."""
        non_header = [_make_assembled_image(location_id=f"s{i}") for i in range(2)]
        opportunities = [
            _make_opportunity(location_id="header", purpose="header"),
        ] + [_make_opportunity(location_id=f"s{i}") for i in range(4)]
        # Target = 4 - 2 = 2, surplus = 2 - 2 = 0
        assert _compute_cuts_count(non_header, opportunities) == 0

    def test_fewer_than_target_no_cuts(self):
        """Fewer surviving images than target → cut 0."""
        non_header = [_make_assembled_image(location_id="s0")]
        opportunities = [
            _make_opportunity(location_id="header", purpose="header"),
        ] + [_make_opportunity(location_id=f"s{i}") for i in range(6)]
        # Target = 6 - 2 = 4, but only 1 survived
        assert _compute_cuts_count(non_header, opportunities) == 0

    def test_one_surplus_cuts_one(self):
        """5 non-header opps, 4 surviving → target=3, surplus=1 → cut 1."""
        non_header = [_make_assembled_image(location_id=f"s{i}") for i in range(4)]
        opportunities = [
            _make_opportunity(location_id="header", purpose="header"),
        ] + [_make_opportunity(location_id=f"s{i}") for i in range(5)]
        # Target = 5 - 2 = 3, surplus = 4 - 3 = 1
        assert _compute_cuts_count(non_header, opportunities) == 1

    def test_no_images_no_cuts(self):
        assert _compute_cuts_count([], []) == 0


# ---------------------------------------------------------------------------
# assemble_document_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAssembleDocumentNode:
    async def test_zero_winners(self):
        state = {
            "input": {"markdown_document": "# Title\n\nSome text"},
            "generation_results": [],
            "selection_results": [],
            "image_plan": [],
            "image_opportunities": [],
        }
        result = await assemble_document_node(state)
        assert result["assembled_images"] == []

    async def test_three_winners(self):
        plans = [_make_plan(location_id=f"s{i}", insertion_after_header=f"Section {i}") for i in range(3)]
        gen_results = [
            _make_gen_result(
                location_id=f"s{i}",
                brief_id=f"s{i}_1",
                image_bytes=b"\x89PNG\r\n\x1a\n" + bytes(i),
            )
            for i in range(3)
        ]
        selections = [
            LocationSelection(
                location_id=f"s{i}",
                selected_brief_id=f"s{i}_1",
                quality_tier="excellent",
                reasoning="good",
            )
            for i in range(3)
        ]
        opportunities = [
            _make_opportunity(location_id=f"s{i}", insertion_after_header=f"Section {i}") for i in range(3)
        ]

        doc = "# Section 0\n\nText 0\n\n# Section 1\n\nText 1\n\n# Section 2\n\nText 2"
        state = {
            "input": {"markdown_document": doc},
            "generation_results": gen_results,
            "selection_results": selections,
            "image_plan": plans,
            "image_opportunities": opportunities,
        }
        result = await assemble_document_node(state)

        assert len(result["assembled_images"]) == 3

    async def test_six_winners(self):
        plans = [_make_plan(location_id=f"s{i}", insertion_after_header=f"Section {i}") for i in range(6)]
        gen_results = [_make_gen_result(location_id=f"s{i}", brief_id=f"s{i}_1") for i in range(6)]
        selections = [
            LocationSelection(
                location_id=f"s{i}",
                selected_brief_id=f"s{i}_1",
                quality_tier="excellent",
                reasoning="good",
            )
            for i in range(6)
        ]
        opportunities = [
            _make_opportunity(location_id=f"s{i}", insertion_after_header=f"Section {i}") for i in range(6)
        ]

        headers = "\n\n".join(f"# Section {i}\n\nText {i}" for i in range(6))
        state = {
            "input": {"markdown_document": headers},
            "generation_results": gen_results,
            "selection_results": selections,
            "image_plan": plans,
            "image_opportunities": opportunities,
        }
        result = await assemble_document_node(state)

        assert len(result["assembled_images"]) == 6

    async def test_includes_purpose_from_opportunities(self):
        plan = _make_plan(location_id="header", insertion_after_header="Title", purpose="header")
        gen_result = _make_gen_result(location_id="header", brief_id="header_1")
        selection = LocationSelection(
            location_id="header",
            selected_brief_id="header_1",
            quality_tier="excellent",
            reasoning="good",
        )
        opportunity = _make_opportunity(location_id="header", purpose="header")

        state = {
            "input": {"markdown_document": "# Title\n\nContent"},
            "generation_results": [gen_result],
            "selection_results": [selection],
            "image_plan": [plan],
            "image_opportunities": [opportunity],
        }
        result = await assemble_document_node(state)

        assert result["assembled_images"][0]["purpose"] == "header"


# ---------------------------------------------------------------------------
# editorial_review_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEditorialReviewNode:
    async def test_no_non_header_images_skips_review(self):
        """When all images are headers, no review needed."""
        state = {
            "assembled_images": [
                _make_assembled_image(location_id="header", purpose="header"),
            ],
            "image_opportunities": [
                _make_opportunity(location_id="header", purpose="header"),
            ],
        }
        result = await editorial_review_node(state)
        assert result["editorial_review_result"]["cut_location_ids"] == []

    async def test_zero_cuts_when_at_target(self):
        """When exactly at target count, cut 0."""
        # 3 non-header opps → target = 1, 1 surviving → surplus = 0
        state = {
            "assembled_images": [
                _make_assembled_image(location_id="s0"),
            ],
            "image_opportunities": [
                _make_opportunity(location_id="header", purpose="header"),
                _make_opportunity(location_id="s0"),
                _make_opportunity(location_id="s1"),
                _make_opportunity(location_id="s2"),
            ],
        }
        result = await editorial_review_node(state)
        assert result["editorial_review_result"]["cut_location_ids"] == []

    async def test_standard_review_with_mocked_llm(self):
        """Standard case: 6 non-header images, cut 2."""
        non_header = [
            _make_assembled_image(location_id=f"s{i}", image_bytes=b"\x89PNG\r\n\x1a\n" + bytes(i)) for i in range(6)
        ]
        # 6 non-header opps → target = 4, 6 surviving → surplus = 2
        opportunities = [
            _make_opportunity(location_id="header", purpose="header"),
        ] + [_make_opportunity(location_id=f"s{i}") for i in range(6)]

        mock_result = EditorialReviewResult(
            evaluations=[
                EditorialImageEvaluation(
                    location_id=f"s{i}",
                    contribution_rank=i + 1,
                    visual_coherence=5 - i % 3,
                    pacing_contribution=4,
                    variety_contribution=3,
                    individual_quality=4,
                    cut_reason="Weakest contribution" if i >= 4 else None,
                )
                for i in range(6)
            ],
            cut_location_ids=["s4", "s5"],
            editorial_summary="Cut two weakest",
        )

        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = mock_result

        mock_base_llm = MagicMock()
        mock_base_llm.with_structured_output.return_value = mock_structured_llm

        state = {
            "assembled_images": non_header,
            "image_opportunities": opportunities,
            "visual_identity": _make_vi(),
        }

        with patch(
            "workflows.output.illustrate.nodes.editorial_review.get_llm",
            return_value=mock_base_llm,
        ):
            result = await editorial_review_node(state)

        assert set(result["editorial_review_result"]["cut_location_ids"]) == {"s4", "s5"}
        assert result["assembled_images"] == []  # Cleared after use

    async def test_header_protection(self):
        """Header images must never appear in cut list."""
        non_header = [
            _make_assembled_image(location_id=f"s{i}", image_bytes=b"\x89PNG\r\n\x1a\nDATA") for i in range(4)
        ]
        header = _make_assembled_image(location_id="header", purpose="header", image_bytes=b"\x89PNG\r\n\x1a\nHDR")
        all_images = [header] + non_header
        # 4 non-header opps → target = 2, 4 surviving → surplus = 2
        opportunities = [
            _make_opportunity(location_id="header", purpose="header"),
        ] + [_make_opportunity(location_id=f"s{i}") for i in range(4)]

        # Mock returns header in cut list (model error) — should be filtered
        mock_result = EditorialReviewResult(
            evaluations=[],
            cut_location_ids=["header", "s3"],
            editorial_summary="Tried to cut header",
        )

        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = mock_result
        mock_base_llm = MagicMock()
        mock_base_llm.with_structured_output.return_value = mock_structured_llm

        state = {
            "assembled_images": all_images,
            "image_opportunities": opportunities,
            "visual_identity": _make_vi(),
        }

        with patch(
            "workflows.output.illustrate.nodes.editorial_review.get_llm",
            return_value=mock_base_llm,
        ):
            result = await editorial_review_node(state)

        cut_ids = result["editorial_review_result"]["cut_location_ids"]
        # Header must NOT be in cut list
        assert "header" not in cut_ids
        # Only valid non-header location should be cut
        assert "s3" in cut_ids

    async def test_llm_failure_keeps_all_images(self):
        """When LLM call fails, keep all images."""
        non_header = [
            _make_assembled_image(location_id=f"s{i}", image_bytes=b"\x89PNG\r\n\x1a\nDATA") for i in range(4)
        ]
        opportunities = [
            _make_opportunity(location_id="header", purpose="header"),
        ] + [_make_opportunity(location_id=f"s{i}") for i in range(6)]

        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.side_effect = Exception("API error")
        mock_base_llm = MagicMock()
        mock_base_llm.with_structured_output.return_value = mock_structured_llm

        state = {
            "assembled_images": non_header,
            "image_opportunities": opportunities,
            "visual_identity": _make_vi(),
        }

        with patch(
            "workflows.output.illustrate.nodes.editorial_review.get_llm",
            return_value=mock_base_llm,
        ):
            result = await editorial_review_node(state)

        assert result["editorial_review_result"]["cut_location_ids"] == []

    async def test_excess_cuts_capped(self):
        """Model returns more cuts than requested — should be capped."""
        non_header = [
            _make_assembled_image(location_id=f"s{i}", image_bytes=b"\x89PNG\r\n\x1a\nDATA") for i in range(6)
        ]
        opportunities = [
            _make_opportunity(location_id="header", purpose="header"),
        ] + [_make_opportunity(location_id=f"s{i}") for i in range(8)]
        # Target = 8 - 2 = 6, surplus = 6 - 6 = 0... wait, need more images
        # Let's fix: 10 opps → target = 8, but 6 surviving → surplus = 0. Not helpful.
        # For surplus=2: need non_header_opps - 2 < len(non_header)
        # 6 non-header opps, 6 surviving → target=4, surplus=2
        opportunities = [
            _make_opportunity(location_id="header", purpose="header"),
        ] + [_make_opportunity(location_id=f"s{i}") for i in range(6)]

        mock_result = EditorialReviewResult(
            evaluations=[],
            cut_location_ids=["s3", "s4", "s5"],  # 3 cuts but only 2 allowed
            editorial_summary="Over-cut",
        )

        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = mock_result
        mock_base_llm = MagicMock()
        mock_base_llm.with_structured_output.return_value = mock_structured_llm

        state = {
            "assembled_images": non_header,
            "image_opportunities": opportunities,
            "visual_identity": _make_vi(),
        }

        with patch(
            "workflows.output.illustrate.nodes.editorial_review.get_llm",
            return_value=mock_base_llm,
        ):
            result = await editorial_review_node(state)

        # Should be capped to 2
        assert len(result["editorial_review_result"]["cut_location_ids"]) == 2

    async def test_oversized_image_skipped_from_vision_call(self):
        """Images exceeding MAX_IMAGE_SIZE are excluded from the vision call."""
        from workflows.output.illustrate.nodes.generate_additional import MAX_IMAGE_SIZE

        oversized_bytes = b"\x89PNG\r\n\x1a\n" + b"X" * (MAX_IMAGE_SIZE + 1)
        normal_bytes = b"\x89PNG\r\n\x1a\nDATA"

        non_header = [
            _make_assembled_image(location_id="s0", image_bytes=normal_bytes),
            _make_assembled_image(location_id="s1", image_bytes=oversized_bytes),
            _make_assembled_image(location_id="s2", image_bytes=normal_bytes),
            _make_assembled_image(location_id="s3", image_bytes=normal_bytes),
        ]
        # 4 non-header opps -> target = 2, 4 surviving -> surplus = 2
        opportunities = [
            _make_opportunity(location_id="header", purpose="header"),
        ] + [_make_opportunity(location_id=f"s{i}") for i in range(4)]

        mock_result = EditorialReviewResult(
            evaluations=[],
            cut_location_ids=["s2", "s3"],
            editorial_summary="Cut two weakest",
        )

        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = mock_result
        mock_base_llm = MagicMock()
        mock_base_llm.with_structured_output.return_value = mock_structured_llm

        state = {
            "assembled_images": non_header,
            "image_opportunities": opportunities,
            "visual_identity": _make_vi(),
        }

        with patch(
            "workflows.output.illustrate.nodes.editorial_review.get_llm",
            return_value=mock_base_llm,
        ):
            result = await editorial_review_node(state)

        # Verify the LLM was called
        mock_structured_llm.ainvoke.assert_called_once()
        call_args = mock_structured_llm.ainvoke.call_args[0][0]
        user_content = call_args[1]["content"]

        # Count image parts -- oversized s1 should be excluded
        image_parts = [p for p in user_content if p.get("type") == "image"]
        assert len(image_parts) == 3  # s0, s2, s3 only (s1 skipped)

        # Verify location labels don't include s1
        text_parts = [
            p
            for p in user_content
            if p.get("type") == "text" and "Image above" in p.get("text", "")
        ]
        location_ids_in_call = [p["text"] for p in text_parts]
        assert not any("s1" in t for t in location_ids_in_call)

        # The review result should still work
        assert result["editorial_review_result"]["cut_location_ids"] == ["s2", "s3"]


# ---------------------------------------------------------------------------
# Finalize filtering editorial cuts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinalizeWithEditorialCuts:
    async def test_filters_out_cut_images(self, tmp_path):
        plans = [_make_plan(location_id=f"s{i}", insertion_after_header=f"Section {i}") for i in range(4)]
        gen_results = [_make_gen_result(location_id=f"s{i}", brief_id=f"s{i}_1") for i in range(4)]
        selections = [
            LocationSelection(
                location_id=f"s{i}",
                selected_brief_id=f"s{i}_1",
                quality_tier="excellent",
                reasoning="good",
            )
            for i in range(4)
        ]

        editorial_result = EditorialReviewResult(
            evaluations=[
                EditorialImageEvaluation(
                    location_id="s2",
                    contribution_rank=4,
                    visual_coherence=2,
                    pacing_contribution=2,
                    variety_contribution=2,
                    individual_quality=2,
                    cut_reason="Weakest image",
                ),
                EditorialImageEvaluation(
                    location_id="s3",
                    contribution_rank=3,
                    visual_coherence=3,
                    pacing_contribution=2,
                    variety_contribution=2,
                    individual_quality=3,
                    cut_reason="Too similar to neighbor",
                ),
            ],
            cut_location_ids=["s2", "s3"],
            editorial_summary="Cut two weakest",
        )

        doc = "\n\n".join(f"# Section {i}\n\nText {i}" for i in range(4))
        state = {
            "input": {"markdown_document": doc},
            "config": IllustrateConfig(output_dir=str(tmp_path)),
            "image_plan": plans,
            "generation_results": gen_results,
            "selection_results": selections,
            "errors": [],
            "editorial_review_result": editorial_result.model_dump(),
        }

        result = await finalize_node(state)

        final_ids = {img["location_id"] for img in result["final_images"]}
        assert "s0" in final_ids
        assert "s1" in final_ids
        assert "s2" not in final_ids
        assert "s3" not in final_ids

    async def test_no_editorial_result_keeps_all(self, tmp_path):
        """When editorial_review_result is absent, keep all images."""
        plans = [_make_plan(location_id="s0", insertion_after_header="Section 0")]
        gen_results = [_make_gen_result(location_id="s0", brief_id="s0_1")]
        selections = [
            LocationSelection(
                location_id="s0",
                selected_brief_id="s0_1",
                quality_tier="excellent",
                reasoning="good",
            ),
        ]

        state = {
            "input": {"markdown_document": "# Section 0\n\nText"},
            "config": IllustrateConfig(output_dir=str(tmp_path)),
            "image_plan": plans,
            "generation_results": gen_results,
            "selection_results": selections,
            "errors": [],
        }

        result = await finalize_node(state)
        assert len(result["final_images"]) == 1

    async def test_editorial_cuts_do_not_cause_false_partial(self, tmp_path):
        """All non-cut locations have images -> status must be 'success', not 'partial'."""
        plans = [_make_plan(location_id=f"s{i}", insertion_after_header=f"Section {i}") for i in range(4)]
        gen_results = [_make_gen_result(location_id=f"s{i}", brief_id=f"s{i}_1") for i in range(4)]
        selections = [
            LocationSelection(
                location_id=f"s{i}",
                selected_brief_id=f"s{i}_1",
                quality_tier="excellent",
                reasoning="good",
            )
            for i in range(4)
        ]

        editorial_result = EditorialReviewResult(
            evaluations=[
                EditorialImageEvaluation(
                    location_id="s2",
                    contribution_rank=4,
                    visual_coherence=2,
                    pacing_contribution=2,
                    variety_contribution=2,
                    individual_quality=2,
                    cut_reason="Weakest image",
                ),
            ],
            cut_location_ids=["s2"],
            editorial_summary="Cut one weakest",
        )

        doc = "\n\n".join(f"# Section {i}\n\nText {i}" for i in range(4))
        state = {
            "input": {"markdown_document": doc},
            "config": IllustrateConfig(output_dir=str(tmp_path)),
            "image_plan": plans,
            "generation_results": gen_results,
            "selection_results": selections,
            "errors": [],
            "editorial_review_result": editorial_result.model_dump(),
        }

        result = await finalize_node(state)

        # 3 out of 4 survived, but 1 was cut — so all expected locations have images
        assert len(result["final_images"]) == 3
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# _determine_status with editorial cuts
# ---------------------------------------------------------------------------


class TestDetermineStatusWithCuts:
    def test_cuts_excluded_from_expected(self):
        """Cutting 1 of 4 planned, 3 images produced -> success."""
        plans = [_make_plan(location_id=f"s{i}") for i in range(4)]
        finals = [
            FinalImage(
                location_id=f"s{i}",
                insertion_after_header=f"Section {i}",
                file_path=f"/tmp/s{i}.png",
                alt_text="img",
                image_type="generated",
                attribution=None,
            )
            for i in [0, 1, 3]
        ]
        assert _determine_status(plans, finals, [], cut_location_ids={"s2"}) == "success"

    def test_no_cuts_still_partial(self):
        """Without cuts, 3 of 4 planned is partial."""
        plans = [_make_plan(location_id=f"s{i}") for i in range(4)]
        finals = [
            FinalImage(
                location_id=f"s{i}",
                insertion_after_header=f"Section {i}",
                file_path=f"/tmp/s{i}.png",
                alt_text="img",
                image_type="generated",
                attribution=None,
            )
            for i in [0, 1, 3]
        ]
        assert _determine_status(plans, finals, []) == "partial"

    def test_all_cut_with_some_remaining(self):
        """All planned cut except those with images -> success."""
        plans = [_make_plan(location_id=f"s{i}") for i in range(4)]
        finals = [
            FinalImage(
                location_id="s0",
                insertion_after_header="Section 0",
                file_path="/tmp/s0.png",
                alt_text="img",
                image_type="generated",
                attribution=None,
            ),
        ]
        assert _determine_status(plans, finals, [], cut_location_ids={"s1", "s2", "s3"}) == "success"

    def test_none_cuts_backward_compat(self):
        """None cut_location_ids behaves like pre-fix code."""
        plans = [_make_plan(location_id=f"s{i}") for i in range(3)]
        finals = [
            FinalImage(
                location_id=f"s{i}",
                insertion_after_header=f"Section {i}",
                file_path=f"/tmp/s{i}.png",
                alt_text="img",
                image_type="generated",
                attribution=None,
            )
            for i in range(3)
        ]
        assert _determine_status(plans, finals, [], cut_location_ids=None) == "success"


# ---------------------------------------------------------------------------
# Graph routing
# ---------------------------------------------------------------------------


class TestRouteAfterSelection:
    def test_no_failures_goes_to_assemble_document(self):
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
        assert route_after_selection(state) == "assemble_document"


class TestRouteAfterAssembly:
    def test_editorial_review_enabled(self):
        state = {"config": IllustrateConfig(enable_editorial_review=True)}
        assert route_after_assembly(state) == "editorial_review"

    def test_editorial_review_disabled(self):
        state = {"config": IllustrateConfig(enable_editorial_review=False)}
        assert route_after_assembly(state) == "finalize"

    def test_default_config_enables_review(self):
        state = {}
        assert route_after_assembly(state) == "editorial_review"
