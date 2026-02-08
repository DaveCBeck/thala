"""Tests for creative_direction_node and plan_briefs_node (mocked LLM calls)."""

from unittest.mock import AsyncMock, patch

import pytest

from workflows.output.illustrate.config import IllustrateConfig
from workflows.output.illustrate.nodes.creative_direction import (
    _extract_title_from_markdown,
    creative_direction_node,
)
from workflows.output.illustrate.nodes.plan_briefs import plan_briefs_node
from workflows.output.illustrate.schemas import (
    CandidateBrief,
    CreativeDirectionResult,
    ImageOpportunity,
    PlanBriefsResult,
    VisualIdentity,
)


class TestExtractTitleFromMarkdown:
    """Test _extract_title_from_markdown helper."""

    def test_extracts_h1(self):
        assert _extract_title_from_markdown("# My Title\nContent") == "My Title"

    def test_skips_frontmatter(self):
        md = "---\ntitle: meta\n---\n# Actual Title\nContent"
        assert _extract_title_from_markdown(md) == "Actual Title"

    def test_falls_back_to_first_line(self):
        assert _extract_title_from_markdown("Just text\nMore text") == "Just text"

    def test_empty_returns_untitled(self):
        assert _extract_title_from_markdown("") == "Untitled Document"


def _make_vi():
    return VisualIdentity(
        primary_style="editorial watercolor",
        color_palette=["warm amber", "deep teal"],
        mood="contemplative",
        lighting="soft diffused",
        avoid=["neon colors"],
    )


def _make_opp(location_id="header", purpose="header", strength="strong"):
    return ImageOpportunity(
        location_id=location_id,
        insertion_after_header="Title",
        purpose=purpose,
        suggested_type="generated",
        strength=strength,
        rationale="Test",
    )


def _make_creative_direction_result():
    return CreativeDirectionResult(
        document_title="Test Article",
        visual_identity=_make_vi(),
        image_opportunities=[
            _make_opp("header", "header"),
            _make_opp("s1", "illustration"),
            _make_opp("s2", "illustration"),
            _make_opp("s3", "illustration", "stretch"),
            _make_opp("s4", "diagram", "stretch"),
        ],
        editorial_notes="Use variety across types",
    )


def _make_plan_briefs_result():
    return PlanBriefsResult(
        candidate_briefs=[
            CandidateBrief(
                location_id="header",
                candidate_index=1,
                image_type="generated",
                brief="A sweeping watercolor landscape",
                relationship_to_text="evocative",
                visual_identity_references="warm amber palette",
            ),
            CandidateBrief(
                location_id="header",
                candidate_index=2,
                image_type="public_domain",
                brief="Sunlit valley photograph",
                relationship_to_text="metaphorical",
                visual_identity_references="soft diffused lighting",
                literal_queries=["valley landscape"],
                conceptual_queries=["serenity dawn"],
                query_strategy="both",
            ),
            CandidateBrief(
                location_id="s1",
                candidate_index=1,
                image_type="diagram",
                brief="Process flow diagram",
                relationship_to_text="explanatory",
                visual_identity_references="teal accents",
                diagram_subtype="flowchart",
            ),
        ],
        brief_strategy_notes="Mixed types for variety",
    )


@pytest.mark.asyncio
class TestCreativeDirectionNode:
    """Test creative_direction_node with mocked invoke."""

    async def test_success(self):
        state = {
            "input": {
                "markdown_document": "# Test\nContent here",
                "title": None,
                "output_dir": None,
            },
            "config": IllustrateConfig(
                generate_header_image=True,
                additional_image_count=2,
            ),
        }

        with patch(
            "workflows.output.illustrate.nodes.creative_direction.invoke",
            new_callable=AsyncMock,
            return_value=_make_creative_direction_result(),
        ):
            result = await creative_direction_node(state)

        assert result["extracted_title"] == "Test Article"
        assert result["visual_identity"].primary_style == "editorial watercolor"
        assert len(result["image_opportunities"]) == 5
        assert result["editorial_notes"] == "Use variety across types"
        # palette_hex should be resolved
        assert len(result["visual_identity"].palette_hex) > 0

    async def test_uses_provided_title(self):
        state = {
            "input": {
                "markdown_document": "Content",
                "title": "Provided Title",
                "output_dir": None,
            },
        }

        mock_invoke = AsyncMock(return_value=_make_creative_direction_result())
        with patch(
            "workflows.output.illustrate.nodes.creative_direction.invoke",
            mock_invoke,
        ):
            await creative_direction_node(state)

        # Check the user prompt contains the provided title
        call_kwargs = mock_invoke.call_args.kwargs
        assert "Provided Title" in call_kwargs["user"]

    async def test_failure_returns_error_state(self):
        state = {
            "input": {
                "markdown_document": "# Test\nContent",
                "title": None,
                "output_dir": None,
            },
        }

        with patch(
            "workflows.output.illustrate.nodes.creative_direction.invoke",
            new_callable=AsyncMock,
            side_effect=Exception("LLM timeout"),
        ):
            result = await creative_direction_node(state)

        assert result["status"] == "failed"
        assert result["image_plan"] == []
        assert len(result["errors"]) == 1
        assert "Creative direction analysis failed" in result["errors"][0]["message"]


@pytest.mark.asyncio
class TestPlanBriefsNode:
    """Test plan_briefs_node with mocked invoke."""

    async def test_success(self):
        state = {
            "input": {
                "markdown_document": "# Test\nContent here",
                "title": None,
                "output_dir": None,
            },
            "config": IllustrateConfig(
                generate_header_image=True,
                additional_image_count=2,
            ),
            "visual_identity": _make_vi(),
            "image_opportunities": [
                _make_opp("header", "header"),
                _make_opp("s1", "illustration"),
                _make_opp("s2", "illustration"),
            ],
            "editorial_notes": "Use variety",
        }

        with patch(
            "workflows.output.illustrate.nodes.plan_briefs.invoke",
            new_callable=AsyncMock,
            return_value=_make_plan_briefs_result(),
        ):
            result = await plan_briefs_node(state)

        assert len(result["candidate_briefs"]) == 3
        assert len(result["image_plan"]) == 2  # Only primary briefs (index=1)
        assert result["image_plan"][0].location_id == "header"
        assert result["image_plan"][1].location_id == "s1"

    async def test_failure_returns_error_state(self):
        state = {
            "input": {
                "markdown_document": "# Test\nContent",
                "title": None,
                "output_dir": None,
            },
            "visual_identity": _make_vi(),
            "image_opportunities": [_make_opp("header", "header")],
            "editorial_notes": "Notes",
        }

        with patch(
            "workflows.output.illustrate.nodes.plan_briefs.invoke",
            new_callable=AsyncMock,
            side_effect=Exception("LLM timeout"),
        ):
            result = await plan_briefs_node(state)

        assert result["status"] == "failed"
        assert result["image_plan"] == []
        assert len(result["errors"]) == 1
        assert "Brief planning failed" in result["errors"][0]["message"]

    async def test_header_pd_override_in_plan(self):
        """When header_prefer_public_domain=True, image_plan header gets public_domain type."""
        state = {
            "input": {
                "markdown_document": "Content",
                "title": None,
                "output_dir": None,
            },
            "config": IllustrateConfig(
                generate_header_image=True,
                additional_image_count=0,
                header_prefer_public_domain=True,
            ),
            "visual_identity": _make_vi(),
            "image_opportunities": [_make_opp("header", "header")],
            "editorial_notes": "",
        }

        # Brief says generated but config prefers PD for header
        result_data = PlanBriefsResult(
            candidate_briefs=[
                CandidateBrief(
                    location_id="header",
                    candidate_index=1,
                    image_type="generated",
                    brief="AI image",
                    relationship_to_text="evocative",
                    visual_identity_references="palette",
                ),
            ],
            brief_strategy_notes="",
        )

        with patch(
            "workflows.output.illustrate.nodes.plan_briefs.invoke",
            new_callable=AsyncMock,
            return_value=result_data,
        ):
            result = await plan_briefs_node(state)

        assert result["image_plan"][0].image_type == "public_domain"
