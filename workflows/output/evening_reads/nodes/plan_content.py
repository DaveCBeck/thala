"""Content planning node using structured output."""

import logging
from typing import Any

from langsmith import traceable

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import ModelTier, invoke, InvokeConfig

from ..schemas import PlanningOutput
from ..prompts import EDITORIAL_STANCE_SECTION, PLANNING_SYSTEM_PROMPT, PLANNING_USER_TEMPLATE
from ..state import DeepDiveAssignment, EveningReadsState

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="EveningReads_PlanContent")
async def plan_content_node(state: EveningReadsState) -> dict[str, Any]:
    """Plan the 4-part series using structured output.

    Uses OPUS for critical decision-making about topic division.

    Returns:
        State update with deep_dive_assignments and overview_scope
    """
    lit_review = state["input"]["literature_review"]
    editorial_stance = state["input"].get("editorial_stance", "")
    citation_keys = state.get("extracted_citation_keys", [])
    citation_mappings = state.get("citation_mappings", {})

    # Group citation keys by recency for the planner
    recent_lines = []  # 2025+
    older_lines = []  # pre-2025 or unknown year
    for key in citation_keys:
        mapping = citation_mappings.get(key, {})
        year = mapping.get("year")
        title = mapping.get("title")
        parts = [key]
        if year:
            parts.append(f"({year})")
        if title:
            parts.append(f"— {title}")
        line = " ".join(parts)
        if year and year >= 2025:
            recent_lines.append(line)
        else:
            older_lines.append(line)

    sections = []
    if recent_lines:
        sections.append(f"### Recent (2025-2026) — {len(recent_lines)} sources\n" + "\n".join(recent_lines))
    if older_lines:
        sections.append(f"### Older (pre-2025) — {len(older_lines)} sources\n" + "\n".join(older_lines))
    citation_keys_str = "\n\n".join(sections) if sections else "None found"

    user_prompt = PLANNING_USER_TEMPLATE.format(
        literature_review=lit_review,
        citation_keys=citation_keys_str,
    )

    # Build system prompt with optional editorial stance
    system_prompt = PLANNING_SYSTEM_PROMPT
    if editorial_stance:
        system_prompt += EDITORIAL_STANCE_SECTION.format(editorial_stance=editorial_stance)
        logger.info("Planning with editorial stance context")

    logger.info(f"Planning series with {len(citation_keys)} available citation keys")

    try:
        planning_result: PlanningOutput = await invoke(
            tier=ModelTier.OPUS,
            system=system_prompt,
            user=user_prompt,
            schema=PlanningOutput,
            config=InvokeConfig(
                max_tokens=4096,
                batch_policy=BatchPolicy.PREFER_BALANCE,
            ),
        )

        # Build recency lookup and pool of recent keys
        recent_keys = {
            k for k in citation_keys
            if citation_mappings.get(k, {}).get("year", 0) >= 2025
        }
        # Track which recent keys are already assigned as anchors
        used_recent_keys: set[str] = set()

        # Convert to state format with recency validation
        deep_dive_assignments: list[DeepDiveAssignment] = []
        for dd in planning_result.deep_dives:
            # Validate anchor keys exist in our citation list
            valid_anchors = [k for k in dd.anchor_keys if k in citation_keys]
            if len(valid_anchors) < len(dd.anchor_keys):
                invalid = set(dd.anchor_keys) - set(valid_anchors)
                logger.warning(
                    f"Deep-dive {dd.id} has invalid anchor keys: {invalid}. Using only valid keys: {valid_anchors}"
                )

            anchors = valid_anchors if valid_anchors else dd.anchor_keys[:3]

            # Recency gate: ensure at least 2 anchors are from 2025+
            recent_anchors = [k for k in anchors if k in recent_keys]
            if len(recent_anchors) < 2 and editorial_stance:
                # Substitute: keep any recent anchors, fill remaining slots from unused recent keys
                needed = 2 - len(recent_anchors)
                available = sorted(
                    recent_keys - used_recent_keys - set(anchors),
                    key=lambda k: citation_mappings.get(k, {}).get("year", 0),
                    reverse=True,
                )
                substitutes = available[:needed]
                if substitutes:
                    old_anchors = anchors
                    anchors = recent_anchors + substitutes + [
                        k for k in anchors if k not in recent_keys
                    ]
                    anchors = anchors[:3]  # Keep max 3
                    logger.warning(
                        f"Deep-dive {dd.id} recency gate: swapped anchors "
                        f"{old_anchors} -> {anchors} (added {substitutes})"
                    )

            used_recent_keys.update(k for k in anchors if k in recent_keys)

            deep_dive_assignments.append(
                DeepDiveAssignment(
                    id=dd.id,
                    title=dd.title,
                    theme=dd.theme,
                    structural_approach=dd.structural_approach,
                    anchor_keys=anchors,
                    relevant_sections=dd.relevant_sections,
                )
            )

        logger.info(f"Planned series: {[(dd['title'], dd['structural_approach']) for dd in deep_dive_assignments]}")

        return {
            "deep_dive_assignments": deep_dive_assignments,
            "overview_scope": planning_result.overview_scope,
        }

    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return {
            "errors": [{"node": "plan_content", "error": str(e)}],
            "status": "failed",
        }
