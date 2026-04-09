"""Replan content node for hook-to-topic feedback loop.

When find_right_now discovers that a deep-dive topic has zero recent hooks,
this node gives the planner a chance to adjust the topic. If the planner
can't adjust without departing from the lit review, it's OK to proceed
without hooks.
"""

import logging
from typing import Any

from langsmith import traceable

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

from ..prompts import EDITORIAL_STANCE_SECTION, PLANNING_SYSTEM_PROMPT, PLANNING_USER_TEMPLATE, REPLAN_FEEDBACK_SECTION
from ..schemas import PlanningOutput
from ..state import DeepDiveAssignment, EveningReadsState

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="EveningReads_ReplanContent")
async def replan_content_node(state: EveningReadsState) -> dict[str, Any]:
    """Replan topics based on hook availability feedback.

    Called when sync_before_write detects deep-dives with zero hooks.
    Gives the planner feedback about which topics had no recent findings
    and asks it to adjust (or keep the topic with acknowledgment).

    Returns:
        Updated deep_dive_assignments and incremented replan_attempts
    """
    assignments = state.get("deep_dive_assignments", [])
    right_now_hooks = state.get("right_now_hooks", [])
    lit_review = state["input"]["literature_review"]
    editorial_stance = state["input"].get("editorial_stance", "")
    editorial_emphasis = state["input"].get("editorial_emphasis", {})
    wants_recency = editorial_emphasis.get("recency") == "high"
    citation_keys = state.get("extracted_citation_keys", [])
    citation_mappings = state.get("citation_mappings", {})

    # Identify topics with zero hooks
    hooks_by_id: dict[str, int] = {a["id"]: 0 for a in assignments}
    for hook in right_now_hooks:
        dd_id = hook.get("deep_dive_id")
        if dd_id in hooks_by_id:
            hooks_by_id[dd_id] += 1

    problem_topics = [
        a for a in assignments if hooks_by_id.get(a["id"], 0) == 0
    ]

    if not problem_topics:
        logger.info("Replan called but all topics have hooks — skipping")
        return {"replan_attempts": state.get("replan_attempts", 0) + 1}

    logger.info(f"Replanning: {len(problem_topics)} topics have zero recent hooks")

    # Build feedback message
    problem_descriptions = []
    for a in problem_topics:
        problem_descriptions.append(f"- {a['id']}: \"{a['title']}\" — {a['theme']}")

    ok_topics = [a for a in assignments if hooks_by_id.get(a["id"], 0) > 0]
    ok_descriptions = []
    for a in ok_topics:
        count = hooks_by_id[a["id"]]
        ok_descriptions.append(f"- {a['id']}: \"{a['title']}\" ({count} hooks found) ✓")

    feedback = (
        "Topics with recent hooks (keep these):\n"
        + "\n".join(ok_descriptions)
        + "\n\nTopics with ZERO recent hooks (consider adjusting):\n"
        + "\n".join(problem_descriptions)
    )

    # Format citation keys (reuse the grouping logic)
    recent_lines = []
    older_lines = []
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

    # Build prompts
    system_prompt = PLANNING_SYSTEM_PROMPT + REPLAN_FEEDBACK_SECTION.format(feedback=feedback)
    if editorial_stance:
        system_prompt += EDITORIAL_STANCE_SECTION.format(editorial_stance=editorial_stance)

    user_prompt = PLANNING_USER_TEMPLATE.format(
        literature_review=lit_review,
        citation_keys=citation_keys_str,
    )

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

        # Apply the same recency gate as plan_content
        import datetime

        current_year = datetime.date.today().year
        recent_keys = {
            k for k in citation_keys
            if (citation_mappings.get(k, {}).get("year") or 0) >= 2025
        }
        current_year_keys = {
            k for k in citation_keys
            if (citation_mappings.get(k, {}).get("year") or 0) >= current_year
        }
        used_recent_keys: set[str] = set()

        deep_dive_assignments: list[DeepDiveAssignment] = []
        for dd in planning_result.deep_dives:
            valid_anchors = [k for k in dd.anchor_keys if k in citation_keys]
            anchors = valid_anchors if valid_anchors else dd.anchor_keys[:3]

            # Recency gate (same as plan_content)
            recent_anchors = [k for k in anchors if k in recent_keys]
            current_year_anchors = [k for k in anchors if k in current_year_keys]
            needs_fix = (len(recent_anchors) < 2 or len(current_year_anchors) < 1) and wants_recency

            if needs_fix:
                available = sorted(
                    recent_keys - used_recent_keys - set(anchors),
                    key=lambda k: citation_mappings.get(k, {}).get("year") or 0,
                    reverse=True,
                )
                if len(current_year_anchors) < 1:
                    cy_available = [k for k in available if k in current_year_keys]
                    if cy_available:
                        sub = cy_available[0]
                        available.remove(sub)
                        anchors = [sub] + anchors[:2]
                        current_year_anchors = [k for k in anchors if k in current_year_keys]

                recent_anchors = [k for k in anchors if k in recent_keys]
                if len(recent_anchors) < 2:
                    needed = 2 - len(recent_anchors)
                    substitutes = available[:needed]
                    if substitutes:
                        anchors = recent_anchors + substitutes + [
                            k for k in anchors if k not in recent_keys
                        ]
                        anchors = anchors[:3]

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

        logger.info(f"Replanned series: {[(dd['title'], dd['structural_approach']) for dd in deep_dive_assignments]}")

        return {
            "deep_dive_assignments": deep_dive_assignments,
            "overview_scope": planning_result.overview_scope,
            "replan_attempts": state.get("replan_attempts", 0) + 1,
        }

    except Exception as e:
        logger.error(f"Replanning failed: {e}")
        return {
            "errors": [{"node": "replan_content", "error": str(e)}],
            "replan_attempts": state.get("replan_attempts", 0) + 1,
        }
