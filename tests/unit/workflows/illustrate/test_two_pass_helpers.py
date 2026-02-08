"""Tests for two-pass planning helpers: _select_opportunities, _briefs_to_image_plan, resolve_palette_hex, build_visual_identity_context."""

from workflows.output.illustrate.config import IllustrateConfig
from workflows.output.illustrate.nodes.plan_briefs import (
    _briefs_to_image_plan,
    _select_opportunities,
)
from workflows.output.illustrate.prompts import (
    build_visual_identity_context,
    resolve_palette_hex,
)
from workflows.output.illustrate.schemas import (
    CandidateBrief,
    ImageOpportunity,
    VisualIdentity,
)


def _make_opp(location_id, purpose="illustration", strength="strong", **kw):
    defaults = dict(
        location_id=location_id,
        insertion_after_header=f"Section {location_id}",
        purpose=purpose,
        suggested_type="generated",
        strength=strength,
        rationale="Test",
    )
    defaults.update(kw)
    return ImageOpportunity(**defaults)


def _make_brief(location_id, candidate_index=1, **kw):
    defaults = dict(
        location_id=location_id,
        candidate_index=candidate_index,
        image_type="generated",
        brief="A test brief",
        relationship_to_text="metaphorical",
        visual_identity_references="Uses palette",
    )
    defaults.update(kw)
    return CandidateBrief(**defaults)


def _make_vi(**kw):
    defaults = dict(
        primary_style="editorial",
        color_palette=["warm amber", "deep teal"],
        mood="calm",
        lighting="natural",
        avoid=["neon colors"],
    )
    defaults.update(kw)
    return VisualIdentity(**defaults)


class TestSelectOpportunities:
    """Test _select_opportunities logic."""

    def test_selects_header_first(self):
        opps = [
            _make_opp("header", purpose="header"),
            _make_opp("s1"),
            _make_opp("s2"),
        ]
        config = IllustrateConfig(generate_header_image=True, additional_image_count=2)
        selected = _select_opportunities(opps, 3, config)
        assert len(selected) == 3
        assert selected[0].location_id == "header"

    def test_prefers_strong_over_stretch(self):
        opps = [
            _make_opp("s1", strength="stretch"),
            _make_opp("s2", strength="strong"),
            _make_opp("s3", strength="strong"),
        ]
        config = IllustrateConfig(generate_header_image=False, additional_image_count=2)
        selected = _select_opportunities(opps, 2, config)
        assert len(selected) == 2
        ids = {o.location_id for o in selected}
        assert "s2" in ids
        assert "s3" in ids

    def test_fills_with_stretch_when_needed(self):
        opps = [
            _make_opp("s1", strength="strong"),
            _make_opp("s2", strength="stretch"),
        ]
        config = IllustrateConfig(generate_header_image=False, additional_image_count=2)
        selected = _select_opportunities(opps, 2, config)
        assert len(selected) == 2
        ids = [o.location_id for o in selected]
        assert ids == ["s1", "s2"]

    def test_no_header_when_disabled(self):
        opps = [
            _make_opp("header", purpose="header"),
            _make_opp("s1"),
        ]
        config = IllustrateConfig(generate_header_image=False, additional_image_count=1)
        selected = _select_opportunities(opps, 1, config)
        assert len(selected) == 1
        assert selected[0].location_id == "s1"

    def test_caps_at_target_count(self):
        opps = [_make_opp(f"s{i}") for i in range(10)]
        config = IllustrateConfig(generate_header_image=False, additional_image_count=3)
        selected = _select_opportunities(opps, 3, config)
        assert len(selected) == 3

    def test_empty_opportunities(self):
        config = IllustrateConfig(generate_header_image=False, additional_image_count=2)
        selected = _select_opportunities([], 2, config)
        assert selected == []


class TestBriefsToImagePlan:
    """Test _briefs_to_image_plan conversion."""

    def test_converts_primary_briefs(self):
        opps = [_make_opp("s1"), _make_opp("s2")]
        briefs = [
            _make_brief("s1", 1),
            _make_brief("s1", 2),
            _make_brief("s2", 1),
        ]
        config = IllustrateConfig()
        plans = _briefs_to_image_plan(briefs, opps, config)
        assert len(plans) == 2
        assert plans[0].location_id == "s1"
        assert plans[1].location_id == "s2"

    def test_skips_candidate_2(self):
        opps = [_make_opp("s1")]
        briefs = [_make_brief("s1", 2)]
        config = IllustrateConfig()
        plans = _briefs_to_image_plan(briefs, opps, config)
        assert len(plans) == 0

    def test_preserves_queries(self):
        opps = [_make_opp("s1", suggested_type="public_domain")]
        briefs = [
            _make_brief(
                "s1",
                image_type="public_domain",
                literal_queries=["forest"],
                conceptual_queries=["peace"],
                query_strategy="both",
            )
        ]
        config = IllustrateConfig()
        plans = _briefs_to_image_plan(briefs, opps, config)
        assert plans[0].literal_queries == ["forest"]
        assert plans[0].conceptual_queries == ["peace"]
        assert plans[0].query_strategy == "both"

    def test_header_pd_override(self):
        opps = [_make_opp("header", purpose="header")]
        briefs = [_make_brief("header", image_type="generated")]
        config = IllustrateConfig(header_prefer_public_domain=True)
        plans = _briefs_to_image_plan(briefs, opps, config)
        assert plans[0].image_type == "public_domain"

    def test_no_header_pd_override_when_disabled(self):
        opps = [_make_opp("header", purpose="header")]
        briefs = [_make_brief("header", image_type="generated")]
        config = IllustrateConfig(header_prefer_public_domain=False)
        plans = _briefs_to_image_plan(briefs, opps, config)
        assert plans[0].image_type == "generated"

    def test_skips_unknown_location(self):
        opps = [_make_opp("s1")]
        briefs = [_make_brief("unknown_location")]
        config = IllustrateConfig()
        plans = _briefs_to_image_plan(briefs, opps, config)
        assert len(plans) == 0

    def test_deduplicates_locations(self):
        opps = [_make_opp("s1")]
        briefs = [
            _make_brief("s1", 1),
            _make_brief("s1", 1),  # duplicate
        ]
        config = IllustrateConfig()
        plans = _briefs_to_image_plan(briefs, opps, config)
        assert len(plans) == 1


class TestResolvePaletteHex:
    """Test resolve_palette_hex color mapping."""

    def test_known_colors(self):
        result = resolve_palette_hex(["warm amber", "deep teal"])
        assert result == ["#F5A623", "#1A6B6A"]

    def test_case_insensitive(self):
        result = resolve_palette_hex(["Warm Amber", "DEEP TEAL"])
        assert result == ["#F5A623", "#1A6B6A"]

    def test_strips_whitespace(self):
        result = resolve_palette_hex(["  warm amber  ", " deep teal"])
        assert result == ["#F5A623", "#1A6B6A"]

    def test_unknown_colors_skipped(self):
        result = resolve_palette_hex(["warm amber", "unicorn sparkle"])
        assert result == ["#F5A623"]

    def test_all_unknown_returns_defaults(self):
        result = resolve_palette_hex(["unicorn sparkle", "dragon breath"])
        assert result == ["#4A90D9", "#7B68EE", "#2E8B57"]

    def test_empty_list_returns_defaults(self):
        result = resolve_palette_hex([])
        assert result == ["#4A90D9", "#7B68EE", "#2E8B57"]


class TestBuildVisualIdentityContext:
    """Test build_visual_identity_context injection helper."""

    def test_none_returns_empty(self):
        assert build_visual_identity_context(None) == ""

    def test_includes_style_and_palette(self):
        vi = _make_vi()
        result = build_visual_identity_context(vi)
        assert "editorial" in result
        assert "warm amber" in result
        assert "deep teal" in result

    def test_includes_avoid_by_default(self):
        vi = _make_vi()
        result = build_visual_identity_context(vi)
        assert "AVOID" in result
        assert "neon colors" in result

    def test_for_imagen_excludes_avoid(self):
        vi = _make_vi()
        result = build_visual_identity_context(vi, for_imagen=True)
        assert "AVOID" not in result
        assert "neon colors" not in result

    def test_for_imagen_includes_style(self):
        vi = _make_vi()
        result = build_visual_identity_context(vi, for_imagen=True)
        assert "editorial" in result
        assert "warm amber" in result
