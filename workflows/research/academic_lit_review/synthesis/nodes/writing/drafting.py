"""Initial draft writing nodes for introduction, methodology, discussion, and conclusions."""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from workflows.shared.language import get_translated_prompt
from ...types import SynthesisState
from .prompts import (
    INTRODUCTION_SYSTEM_PROMPT,
    INTRODUCTION_USER_TEMPLATE,
    METHODOLOGY_SYSTEM_PROMPT,
    METHODOLOGY_USER_TEMPLATE,
    DISCUSSION_SYSTEM_PROMPT,
    DISCUSSION_USER_TEMPLATE,
    CONCLUSIONS_SYSTEM_PROMPT,
    CONCLUSIONS_USER_TEMPLATE,
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

    themes_overview = "\n".join(
        f"- {c['label']}: {c['description'][:100]}" for c in clusters
    )

    years = [s.get("year", 0) for s in paper_summaries.values() if s.get("year")]
    if years:
        actual_range = f"{min(years)}-{max(years)}"
    elif date_range:
        actual_range = f"{date_range[0]}-{date_range[1]}"
    else:
        actual_range = "Not specified"

    intro_system = INTRODUCTION_SYSTEM_PROMPT
    intro_user_template = INTRODUCTION_USER_TEMPLATE
    method_system = METHODOLOGY_SYSTEM_PROMPT
    method_user_template = METHODOLOGY_USER_TEMPLATE

    if language_config and language_config["code"] != "en":
        intro_system = await get_translated_prompt(
            INTRODUCTION_SYSTEM_PROMPT,
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
            METHODOLOGY_SYSTEM_PROMPT,
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

    llm = get_llm(tier=ModelTier.SONNET, max_tokens=4096)

    intro_response = await invoke_with_cache(
        llm,
        system_prompt=intro_system,
        user_prompt=intro_prompt,
    )

    introduction = (
        intro_response.content
        if isinstance(intro_response.content, str)
        else intro_response.content[0].get("text", "")
    )

    total_papers = len(paper_summaries)

    method_prompt = method_user_template.format(
        topic=topic,
        keyword_count=total_papers // 4,
        citation_count=total_papers * 3 // 4,
        total_papers=total_papers,
        processed_count=total_papers,
        max_stages=quality_settings.get("max_stages", 5),
        saturation_threshold=quality_settings.get("saturation_threshold", 0.1),
        min_citations=quality_settings.get("min_citations_filter", 10),
        date_range=actual_range,
        final_corpus_size=total_papers,
        cluster_count=len(clusters),
    )

    method_response = await invoke_with_cache(
        llm,
        system_prompt=method_system,
        user_prompt=method_prompt,
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


async def write_discussion_conclusions_node(state: SynthesisState) -> dict[str, Any]:
    """Write discussion and conclusions sections."""
    input_data = state.get("input", {})
    clusters = state.get("clusters", [])
    cluster_analyses = state.get("cluster_analyses", [])
    language_config = state.get("language_config")

    research_questions = input_data.get("research_questions", [])

    themes_summary = "\n".join(
        f"- {c['label']}: {c['description'][:150]}" for c in clusters
    )

    gaps = []
    for analysis in cluster_analyses:
        for question in analysis.get("outstanding_questions", []):
            gaps.append(question)

    for cluster in clusters:
        for gap in cluster.get("gaps", []):
            gaps.append(gap)

    discussion_system = DISCUSSION_SYSTEM_PROMPT
    discussion_user_template = DISCUSSION_USER_TEMPLATE
    conclusions_system = CONCLUSIONS_SYSTEM_PROMPT
    conclusions_user_template = CONCLUSIONS_USER_TEMPLATE

    if language_config and language_config["code"] != "en":
        discussion_system = await get_translated_prompt(
            DISCUSSION_SYSTEM_PROMPT,
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
            CONCLUSIONS_SYSTEM_PROMPT,
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
        cross_cutting_findings="See thematic sections for detailed findings.",
        research_gaps="\n".join(f"- {g}" for g in gaps[:10])
        or "None explicitly identified",
    )

    llm = get_llm(tier=ModelTier.SONNET, max_tokens=4096)

    discussion_response = await invoke_with_cache(
        llm,
        system_prompt=discussion_system,
        user_prompt=discussion_prompt,
    )

    discussion = (
        discussion_response.content
        if isinstance(discussion_response.content, str)
        else discussion_response.content[0].get("text", "")
    )

    findings_summary = "\n".join(
        f"Q{i + 1}: {q}\n   Finding: Based on {len(clusters)} themes covering {sum(len(c['paper_dois']) for c in clusters)} papers"
        for i, q in enumerate(research_questions)
    )

    conclusions_prompt = conclusions_user_template.format(
        research_questions="\n".join(f"- {q}" for q in research_questions),
        findings_per_question=findings_summary,
        main_contributions=f"Systematic review of {sum(len(c['paper_dois']) for c in clusters)} papers organized into {len(clusters)} themes",
    )

    conclusions_response = await invoke_with_cache(
        llm,
        system_prompt=conclusions_system,
        user_prompt=conclusions_prompt,
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
