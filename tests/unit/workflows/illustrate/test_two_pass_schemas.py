"""Tests for two-pass planning schemas (VisualIdentity, ImageOpportunity, CandidateBrief, etc.)."""

import json

from workflows.output.illustrate.schemas import (
    CandidateBrief,
    CreativeDirectionResult,
    ImageOpportunity,
    PlanBriefsResult,
    VisualIdentity,
)


class TestVisualIdentity:
    """Test VisualIdentity schema and validators."""

    def _make(self, **overrides):
        defaults = dict(
            primary_style="editorial watercolor illustration",
            color_palette=["warm amber", "deep teal", "ivory"],
            mood="contemplative, intellectual",
            lighting="soft diffused natural light",
            avoid=["photorealistic faces", "neon colors"],
        )
        defaults.update(overrides)
        return VisualIdentity(**defaults)

    def test_basic_creation(self):
        vi = self._make()
        assert vi.primary_style == "editorial watercolor illustration"
        assert len(vi.color_palette) == 3
        assert len(vi.avoid) == 2

    def test_palette_hex_default_empty(self):
        vi = self._make()
        assert vi.palette_hex == []

    def test_palette_hex_set(self):
        vi = self._make(palette_hex=["#F5A623", "#1A6B6A"])
        assert vi.palette_hex == ["#F5A623", "#1A6B6A"]

    def test_color_palette_json_string(self):
        json_str = json.dumps(["warm amber", "deep teal"])
        vi = self._make(color_palette=json_str)
        assert vi.color_palette == ["warm amber", "deep teal"]

    def test_avoid_json_string(self):
        json_str = json.dumps(["neon colors", "clipart"])
        vi = self._make(avoid=json_str)
        assert vi.avoid == ["neon colors", "clipart"]

    def test_color_palette_none_becomes_empty(self):
        vi = self._make(color_palette=None)
        assert vi.color_palette == []

    def test_avoid_none_becomes_empty(self):
        vi = self._make(avoid=None)
        assert vi.avoid == []

    def test_palette_hex_json_string(self):
        json_str = json.dumps(["#F5A623", "#1A6B6A"])
        vi = self._make(palette_hex=json_str)
        assert vi.palette_hex == ["#F5A623", "#1A6B6A"]


class TestImageOpportunity:
    """Test ImageOpportunity schema."""

    def _make(self, **overrides):
        defaults = dict(
            location_id="header",
            insertion_after_header="Introduction",
            purpose="header",
            suggested_type="generated",
            strength="strong",
            rationale="Sets emotional tone",
        )
        defaults.update(overrides)
        return ImageOpportunity(**defaults)

    def test_basic_creation(self):
        opp = self._make()
        assert opp.location_id == "header"
        assert opp.strength == "strong"
        assert opp.diagram_subtype is None

    def test_with_diagram_subtype(self):
        opp = self._make(
            purpose="diagram",
            suggested_type="diagram",
            diagram_subtype="flowchart",
        )
        assert opp.diagram_subtype == "flowchart"

    def test_stretch_strength(self):
        opp = self._make(strength="stretch")
        assert opp.strength == "stretch"


class TestCandidateBrief:
    """Test CandidateBrief schema and validators."""

    def _make(self, **overrides):
        defaults = dict(
            location_id="section_1",
            candidate_index=1,
            image_type="generated",
            brief="A sweeping watercolor landscape",
            relationship_to_text="metaphorical",
            visual_identity_references="Uses warm amber palette",
        )
        defaults.update(overrides)
        return CandidateBrief(**defaults)

    def test_basic_creation(self):
        cb = self._make()
        assert cb.location_id == "section_1"
        assert cb.candidate_index == 1
        assert cb.literal_queries == []
        assert cb.conceptual_queries == []

    def test_with_queries(self):
        cb = self._make(
            image_type="public_domain",
            literal_queries=["forest trail"],
            conceptual_queries=["peace serenity"],
            query_strategy="both",
        )
        assert cb.literal_queries == ["forest trail"]
        assert cb.conceptual_queries == ["peace serenity"]
        assert cb.query_strategy == "both"

    def test_queries_json_string(self):
        json_str = json.dumps(["term one", "term two"])
        cb = self._make(literal_queries=json_str, conceptual_queries=json_str)
        assert cb.literal_queries == ["term one", "term two"]
        assert cb.conceptual_queries == ["term one", "term two"]

    def test_queries_none_becomes_empty(self):
        cb = self._make(literal_queries=None, conceptual_queries=None)
        assert cb.literal_queries == []
        assert cb.conceptual_queries == []

    def test_candidate_index_2(self):
        cb = self._make(candidate_index=2)
        assert cb.candidate_index == 2

    def test_with_diagram_subtype(self):
        cb = self._make(
            image_type="diagram",
            diagram_subtype="flowchart",
        )
        assert cb.diagram_subtype == "flowchart"

    def test_relationship_to_text_values(self):
        for val in ("literal", "metaphorical", "explanatory", "evocative"):
            cb = self._make(relationship_to_text=val)
            assert cb.relationship_to_text == val


class TestCreativeDirectionResult:
    """Test CreativeDirectionResult schema."""

    def test_basic_creation(self):
        result = CreativeDirectionResult(
            document_title="Test Article",
            visual_identity=VisualIdentity(
                primary_style="editorial",
                color_palette=["blue", "green"],
                mood="calm",
                lighting="natural",
                avoid=["neon"],
            ),
            image_opportunities=[
                ImageOpportunity(
                    location_id="header",
                    insertion_after_header="Title",
                    purpose="header",
                    suggested_type="generated",
                    strength="strong",
                    rationale="Sets tone",
                ),
            ],
            editorial_notes="Use variety",
        )
        assert result.document_title == "Test Article"
        assert len(result.image_opportunities) == 1

    def test_opportunities_json_string(self):
        opp = {
            "location_id": "header",
            "insertion_after_header": "Title",
            "purpose": "header",
            "suggested_type": "generated",
            "strength": "strong",
            "rationale": "Sets tone",
        }
        result = CreativeDirectionResult(
            document_title="Test",
            visual_identity=VisualIdentity(
                primary_style="editorial",
                color_palette=["blue"],
                mood="calm",
                lighting="natural",
                avoid=[],
            ),
            image_opportunities=json.dumps([opp]),
            editorial_notes="Notes",
        )
        assert len(result.image_opportunities) == 1


class TestPlanBriefsResult:
    """Test PlanBriefsResult schema."""

    def test_basic_creation(self):
        result = PlanBriefsResult(
            candidate_briefs=[
                CandidateBrief(
                    location_id="header",
                    candidate_index=1,
                    image_type="generated",
                    brief="A test brief",
                    relationship_to_text="literal",
                    visual_identity_references="Uses palette",
                ),
            ],
            brief_strategy_notes="Mixed types for variety",
        )
        assert len(result.candidate_briefs) == 1

    def test_briefs_json_string(self):
        brief_dict = {
            "location_id": "header",
            "candidate_index": 1,
            "image_type": "generated",
            "brief": "A test brief",
            "relationship_to_text": "literal",
            "visual_identity_references": "Uses palette",
        }
        result = PlanBriefsResult(
            candidate_briefs=json.dumps([brief_dict]),
            brief_strategy_notes="Notes",
        )
        assert len(result.candidate_briefs) == 1
