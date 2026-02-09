"""Pass 2: Generate candidate briefs for each image opportunity."""

import json
import logging

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

from ..config import IllustrateConfig
from ..prompts import PLAN_BRIEFS_SYSTEM, PLAN_BRIEFS_USER, build_visual_identity_context
from ..schemas import CandidateBrief, ImageLocationPlan, ImageOpportunity, PlanBriefsResult
from ..state import IllustrateState

logger = logging.getLogger(__name__)


def _select_opportunities(
    opportunities: list[ImageOpportunity],
    target_count: int,
    config: IllustrateConfig,
) -> list[ImageOpportunity]:
    """Select opportunities: prefer 'strong' over 'stretch'.

    Always include a header opportunity if config.generate_header_image.
    Fill remaining slots with 'strong' first, then 'stretch'.
    """
    selected: list[ImageOpportunity] = []

    # Separate header from non-header
    header_opps = [o for o in opportunities if o.purpose == "header"]
    non_header_opps = [o for o in opportunities if o.purpose != "header"]

    # Include header if configured
    if config.generate_header_image and header_opps:
        selected.append(header_opps[0])

    # Fill remaining slots
    remaining = target_count - len(selected)
    strong = [o for o in non_header_opps if o.strength == "strong"]
    stretch = [o for o in non_header_opps if o.strength == "stretch"]

    selected.extend(strong[:remaining])
    remaining = target_count - len(selected)
    if remaining > 0:
        selected.extend(stretch[:remaining])

    return selected


def _briefs_to_image_plan(
    briefs: list[CandidateBrief],
    opportunities: list[ImageOpportunity],
    config: IllustrateConfig,
) -> list[ImageLocationPlan]:
    """Convert candidate briefs to ImageLocationPlan for backward compatibility.

    Uses the primary brief (candidate_index=1) for each location.
    """
    opp_map = {o.location_id: o for o in opportunities}
    plans: list[ImageLocationPlan] = []
    seen_locations: set[str] = set()

    for brief in briefs:
        if brief.candidate_index != 1:
            continue
        if brief.location_id in seen_locations:
            continue
        seen_locations.add(brief.location_id)

        opp = opp_map.get(brief.location_id)
        if not opp:
            continue

        # Override header type if config prefers public domain
        image_type = brief.image_type
        if opp.purpose == "header" and config.header_prefer_public_domain:
            image_type = "public_domain"

        plans.append(
            ImageLocationPlan(
                location_id=brief.location_id,
                insertion_after_header=opp.insertion_after_header,
                purpose=opp.purpose,
                image_type=image_type,
                brief=brief.brief,
                literal_queries=brief.literal_queries,
                conceptual_queries=brief.conceptual_queries,
                query_strategy=brief.query_strategy,
                diagram_subtype=brief.diagram_subtype,
            )
        )

    return plans


async def plan_briefs_node(state: IllustrateState) -> dict:
    """Pass 2: Generate candidate briefs for each image opportunity.

    LLM sees document + visual identity + opportunities from Pass 1.
    Produces up to 2 candidate briefs per location.

    Returns:
        State updates with candidate_briefs and backward-compatible image_plan.
    """
    config = state.get("config") or IllustrateConfig()
    document = state["input"]["markdown_document"]
    visual_identity = state.get("visual_identity")
    image_opportunities = state.get("image_opportunities", [])

    if not visual_identity or not image_opportunities:
        logger.warning("Skipping plan_briefs: no visual identity or opportunities (creative_direction may have failed)")
        return {"image_plan": [], "status": "failed"}

    opportunities: list[ImageOpportunity] = image_opportunities
    editorial_notes = state.get("editorial_notes", "")

    target_count = (1 if config.generate_header_image else 0) + config.additional_image_count
    # Select N+2 to maintain over-generation surplus through to editorial review
    select_count = target_count + 2
    selected = _select_opportunities(opportunities, select_count, config)

    try:
        result = await invoke(
            tier=ModelTier.SONNET,
            system=PLAN_BRIEFS_SYSTEM,
            user=PLAN_BRIEFS_USER.format(
                document=document,
                visual_identity_text=build_visual_identity_context(visual_identity),
                opportunities_text=json.dumps([o.model_dump() for o in selected], indent=2),
                editorial_notes=editorial_notes,
            ),
            schema=PlanBriefsResult,
            config=InvokeConfig(max_tokens=8000, batch_policy=BatchPolicy.PREFER_SPEED),
        )

        image_plan = _briefs_to_image_plan(result.candidate_briefs, selected, config)

        planned_ids = {p.location_id for p in image_plan}
        selected_ids = {o.location_id for o in selected}
        missing = selected_ids - planned_ids
        if missing:
            logger.warning(f"LLM produced no briefs for locations: {missing}")

        logger.info(f"Plan briefs complete: {len(result.candidate_briefs)} briefs across {len(image_plan)} locations")

        return {
            "candidate_briefs": result.candidate_briefs,
            "image_plan": image_plan,
        }

    except Exception as e:
        logger.error(f"Plan briefs failed: {e}")
        return {
            "image_plan": [],
            "errors": [
                {
                    "location_id": None,
                    "severity": "error",
                    "message": "Brief planning failed",
                    "stage": "analysis",
                }
            ],
            "status": "failed",
        }
