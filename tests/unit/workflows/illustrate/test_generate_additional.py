"""Tests for generate_additional multi-query logic (A5)."""

from workflows.output.illustrate.nodes.generate_additional import _build_search_queries
from workflows.output.illustrate.schemas import ImageLocationPlan


def _make_plan(**overrides):
    defaults = dict(
        location_id="section_1",
        insertion_after_header="Introduction",
        purpose="illustration",
        image_type="public_domain",
        brief="A striking image of nature",
    )
    defaults.update(overrides)
    return ImageLocationPlan(**defaults)


class TestBuildSearchQueries:
    """Test _build_search_queries from ImageLocationPlan."""

    def test_empty_when_no_multi_query_fields(self):
        plan = _make_plan()
        assert _build_search_queries(plan) == []

    def test_literal_strategy(self):
        plan = _make_plan(
            literal_queries=["forest trail", "woodland path"],
            conceptual_queries=["peace serenity morning"],
            query_strategy="literal",
        )
        result = _build_search_queries(plan)
        assert result == ["forest trail", "woodland path"]

    def test_conceptual_strategy(self):
        plan = _make_plan(
            literal_queries=["forest trail"],
            conceptual_queries=["peace serenity", "calm dawn"],
            query_strategy="conceptual",
        )
        result = _build_search_queries(plan)
        assert result == ["peace serenity", "calm dawn"]

    def test_both_strategy_interleaves(self):
        plan = _make_plan(
            literal_queries=["forest", "trees"],
            conceptual_queries=["peace", "serenity"],
            query_strategy="both",
        )
        result = _build_search_queries(plan)
        # Interleaved: conceptual first, then literal
        assert result == ["peace", "forest", "serenity", "trees"]

    def test_both_strategy_caps_at_four(self):
        plan = _make_plan(
            literal_queries=["a", "b", "c"],
            conceptual_queries=["x", "y", "z"],
            query_strategy="both",
        )
        result = _build_search_queries(plan)
        assert len(result) <= 4

    def test_default_strategy_is_both(self):
        plan = _make_plan(
            literal_queries=["forest"],
            conceptual_queries=["peace"],
        )
        # query_strategy defaults to None, should behave as "both"
        result = _build_search_queries(plan)
        assert result == ["peace", "forest"]

    def test_only_conceptual_queries_populated(self):
        plan = _make_plan(
            conceptual_queries=["morning light", "renewal"],
        )
        result = _build_search_queries(plan)
        assert result == ["morning light", "renewal"]

    def test_only_literal_queries_populated(self):
        plan = _make_plan(
            literal_queries=["oak tree", "deciduous forest"],
        )
        result = _build_search_queries(plan)
        assert result == ["oak tree", "deciduous forest"]


class TestImageLocationPlanMultiQuery:
    """Test that schema properly has multi-query fields with defaults."""

    def test_backwards_compatible_no_multi_query(self):
        """Old plans without multi-query fields should still work."""
        plan = _make_plan(search_query="sunset beach")
        assert plan.literal_queries == []
        assert plan.conceptual_queries == []
        assert plan.query_strategy is None
        assert plan.search_query == "sunset beach"

    def test_full_multi_query_plan(self):
        plan = _make_plan(
            literal_queries=["autophagy cell biology"],
            conceptual_queries=["renewal spring rebirth"],
            query_strategy="conceptual",
        )
        assert plan.literal_queries == ["autophagy cell biology"]
        assert plan.conceptual_queries == ["renewal spring rebirth"]
        assert plan.query_strategy == "conceptual"
