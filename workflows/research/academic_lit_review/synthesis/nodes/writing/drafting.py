"""Initial draft writing nodes for introduction, methodology, discussion, and conclusions."""

import asyncio
import logging
from typing import Any

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import ModelTier, invoke, InvokeConfig
from workflows.shared.language import get_translated_prompt
from ...types import SynthesisState
from ...transparency import render_transparency_for_prompt
from .prompts import (
    get_introduction_system_prompt,
    INTRODUCTION_USER_TEMPLATE,
    get_methodology_system_prompt,
    METHODOLOGY_USER_TEMPLATE,
    get_discussion_system_prompt,
    DISCUSSION_USER_TEMPLATE,
    get_conclusions_system_prompt,
    CONCLUSIONS_USER_TEMPLATE,
    EDITORIAL_STANCE_SECTION,
    DEFAULT_TARGET_WORDS,
)

logger = logging.getLogger(__name__)


async def write_intro_methodology_node(state: SynthesisState) -> dict[str, Any]:
    """Write introduction and methodology sections."""
    input_data = state.get("input", {})
    paper_summaries = state.get("paper_summaries", {})
    clusters = state.get("clusters", [])
    quality_settings = state.get("quality_settings", {})
    language_config = state.get("language_config")

    topic = input_data.get("topic", "Unknown topic")
    research_questions = input_data.get("research_questions", [])
    date_range = input_data.get("date_range")

    themes_overview = "\n".join(f"- {c['label']}: {c['description'][:100]}" for c in clusters)

    years = [s.get("year", 0) for s in paper_summaries.values() if s.get("year")]
    if years:
        actual_range = f"{min(years)}-{max(years)}"
    elif date_range:
        actual_range = f"{date_range[0]}-{date_range[1]}"
    else:
        actual_range = "Not specified"

    target_words = quality_settings.get("target_word_count", DEFAULT_TARGET_WORDS)
    intro_system = get_introduction_system_prompt(target_words)
    intro_user_template = INTRODUCTION_USER_TEMPLATE
    method_system = get_methodology_system_prompt(target_words)
    method_user_template = METHODOLOGY_USER_TEMPLATE

    if language_config and language_config["code"] != "en":
        intro_system = await get_translated_prompt(
            intro_system,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_intro_system",
        )
        intro_user_template = await get_translated_prompt(
            INTRODUCTION_USER_TEMPLATE,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_intro_user",
        )
        method_system = await get_translated_prompt(
            method_system,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_method_system",
        )
        method_user_template = await get_translated_prompt(
            METHODOLOGY_USER_TEMPLATE,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_method_user",
        )

    intro_prompt = intro_user_template.format(
        topic=topic,
        research_questions="\n".join(f"- {q}" for q in research_questions),
        themes_overview=themes_overview,
        paper_count=len(paper_summaries),
        date_range=actual_range,
    )

    # Build methodology prompt from real transparency data
    transparency_report = state.get("transparency_report")
    if transparency_report:
        prompt_vars = render_transparency_for_prompt(transparency_report)
        prompt_vars["topic"] = topic
        method_prompt = method_user_template.format(**prompt_vars)
    else:
        # Fallback when no transparency report is available (e.g. re-runs against
        # older saved states). The model must NOT invent search details it was not
        # given, so we explicitly authorise a short descriptive placeholder.
        total_papers = len(paper_summaries)
        logger.debug("No transparency_report in state, using basic methodology prompt")
        method_prompt = (
            f"Write a brief methodology placeholder (maximum 120 words, one short paragraph) "
            f"for this literature review on: {topic}\n\n"
            f"AVAILABLE FACTS (use only these):\n"
            f"- Final corpus: {total_papers} papers organised into {len(clusters)} thematic clusters\n"
            f"- Date range of literature: {actual_range}\n\n"
            f"No upstream search log is available in this run. Acknowledge that the search "
            f"and screening record is not reproducible here, describe the corpus size and "
            f"date range honestly, and state that thematic clusters were derived by the "
            f"reviewer from the assembled corpus. Do NOT invent databases, Boolean queries, "
            f"screening counts, or PRISMA steps. Do NOT reach the 450-word ceiling; stay "
            f"well under 120 words."
        )

    intro_coro = invoke(
        tier=ModelTier.SONNET,
        system=intro_system,
        user=intro_prompt,
        config=InvokeConfig(
            max_tokens=4096,
            batch_policy=BatchPolicy.PREFER_BALANCE,
        ),
    )

    method_coro = invoke(
        tier=ModelTier.SONNET,
        system=method_system,
        user=method_prompt,
        config=InvokeConfig(
            max_tokens=4096,
            batch_policy=BatchPolicy.PREFER_BALANCE,
        ),
    )

    intro_response, method_response = await asyncio.gather(intro_coro, method_coro)

    introduction = (
        intro_response.content if isinstance(intro_response.content, str) else intro_response.content[0].get("text", "")
    )

    methodology = (
        method_response.content
        if isinstance(method_response.content, str)
        else method_response.content[0].get("text", "")
    )

    logger.info("Completed introduction and methodology sections")

    return {
        "introduction_draft": introduction,
        "methodology_draft": methodology,
    }


def _format_thematic_content(
    clusters: list[dict],
    thematic_section_drafts: dict[str, str],
) -> str:
    """Render the full thematic section drafts in cluster order for downstream prompts.

    Discussion, conclusions, and abstract all need to see the actual prose that
    was written for each theme — not just cluster labels — in order to synthesize
    across themes rather than restate the topic.
    """
    if not thematic_section_drafts:
        return "[No thematic sections available]"

    parts = []
    for i, cluster in enumerate(clusters, start=1):
        label = cluster["label"]
        body = thematic_section_drafts.get(label, f"[Section for {label} not available]")
        parts.append(f"## Section {i}. {label}\n\n{body}")
    return "\n\n".join(parts)


async def write_discussion_conclusions_node(state: SynthesisState) -> dict[str, Any]:
    """Write discussion and conclusions sections.

    Discussion is written first with full thematic content so it can synthesize
    across themes. Conclusions is written second with both the thematic content
    AND the just-written discussion, so it can build on the synthesis rather
    than compete with it or restate the abstract.
    """
    input_data = state.get("input", {})
    clusters = state.get("clusters", [])
    cluster_analyses = state.get("cluster_analyses", [])
    thematic_section_drafts = state.get("thematic_section_drafts", {})
    language_config = state.get("language_config")
    quality_settings = state.get("quality_settings", {})

    research_questions = input_data.get("research_questions", [])

    themes_summary = "\n".join(f"- {c['label']}: {c['description'][:150]}" for c in clusters)
    thematic_content = _format_thematic_content(clusters, thematic_section_drafts)

    gaps = []
    for analysis in cluster_analyses:
        for question in analysis.get("outstanding_questions", []):
            gaps.append(question)

    for cluster in clusters:
        for gap in cluster.get("gaps", []):
            gaps.append(gap)

    target_words = quality_settings.get("target_word_count", DEFAULT_TARGET_WORDS)
    discussion_system = get_discussion_system_prompt(target_words)
    discussion_user_template = DISCUSSION_USER_TEMPLATE
    conclusions_system = get_conclusions_system_prompt(target_words)
    conclusions_user_template = CONCLUSIONS_USER_TEMPLATE

    if language_config and language_config["code"] != "en":
        discussion_system = await get_translated_prompt(
            discussion_system,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_discussion_system",
        )
        discussion_user_template = await get_translated_prompt(
            DISCUSSION_USER_TEMPLATE,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_discussion_user",
        )
        conclusions_system = await get_translated_prompt(
            conclusions_system,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_conclusions_system",
        )
        conclusions_user_template = await get_translated_prompt(
            CONCLUSIONS_USER_TEMPLATE,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_conclusions_user",
        )

    discussion_prompt = discussion_user_template.format(
        research_questions="\n".join(f"- {q}" for q in research_questions),
        themes_summary=themes_summary,
        thematic_content=thematic_content,
        research_gaps="\n".join(f"- {g}" for g in gaps[:10]) or "None explicitly identified",
    )

    # Append editorial stance to user prompts when present (priors, not mandates)
    editorial_stance = state.get("editorial_stance")
    if editorial_stance:
        stance_block = EDITORIAL_STANCE_SECTION.format(editorial_stance=editorial_stance)
        discussion_prompt += f"\n\n{stance_block}"

    discussion_response = await invoke(
        tier=ModelTier.SONNET,
        system=discussion_system,
        user=discussion_prompt,
        config=InvokeConfig(
            max_tokens=4096,
            batch_policy=BatchPolicy.PREFER_BALANCE,
        ),
    )

    discussion = (
        discussion_response.content
        if isinstance(discussion_response.content, str)
        else discussion_response.content[0].get("text", "")
    )

    conclusions_prompt = conclusions_user_template.format(
        research_questions="\n".join(f"- {q}" for q in research_questions),
        thematic_content=thematic_content,
        discussion=discussion,
    )

    if editorial_stance:
        stance_block = EDITORIAL_STANCE_SECTION.format(editorial_stance=editorial_stance)
        conclusions_prompt += f"\n\n{stance_block}"

    conclusions_response = await invoke(
        tier=ModelTier.SONNET,
        system=conclusions_system,
        user=conclusions_prompt,
        config=InvokeConfig(
            max_tokens=4096,
            batch_policy=BatchPolicy.PREFER_BALANCE,
        ),
    )

    conclusions = (
        conclusions_response.content
        if isinstance(conclusions_response.content, str)
        else conclusions_response.content[0].get("text", "")
    )

    logger.info("Completed discussion and conclusions sections")

    return {
        "discussion_draft": discussion,
        "conclusions_draft": conclusions,
    }
