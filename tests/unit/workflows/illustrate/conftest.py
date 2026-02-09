"""Shared test helper factories for illustrate workflow tests."""

from workflows.output.illustrate.schemas import (
    CandidateBrief,
    ImageLocationPlan,
    ImageOpportunity,
    VisualIdentity,
)
from workflows.output.illustrate.state import (
    AssembledImage,
    ImageGenResult,
)


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


def _make_opportunity(**overrides):
    defaults = dict(
        location_id="section_1",
        insertion_after_header="Introduction",
        purpose="illustration",
        suggested_type="generated",
        strength="strong",
        rationale="Helps readers visualize the concept",
    )
    defaults.update(overrides)
    return ImageOpportunity(**defaults)


def _make_vi():
    return VisualIdentity(
        primary_style="editorial watercolor",
        color_palette=["warm amber", "deep teal"],
        mood="contemplative",
        lighting="soft diffused",
        avoid=["neon colors"],
    )


def _make_assembled_image(**overrides):
    defaults = dict(
        location_id="section_1",
        image_type="generated",
        purpose="illustration",
        image_bytes=b"IMG_BYTES",
    )
    defaults.update(overrides)
    return AssembledImage(**defaults)
