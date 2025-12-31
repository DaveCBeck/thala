"""Writing nodes for synthesis subgraph."""

import asyncio
import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from ..types import SynthesisState, MAX_CONCURRENT_SECTIONS
from ..prompts import (
    INTRODUCTION_SYSTEM_PROMPT,
    INTRODUCTION_USER_TEMPLATE,
    METHODOLOGY_SYSTEM_PROMPT,
    METHODOLOGY_USER_TEMPLATE,
    THEMATIC_SECTION_SYSTEM_PROMPT,
    THEMATIC_SECTION_USER_TEMPLATE,
    DISCUSSION_SYSTEM_PROMPT,
    DISCUSSION_USER_TEMPLATE,
    CONCLUSIONS_SYSTEM_PROMPT,
    CONCLUSIONS_USER_TEMPLATE,
)
from ..citation_utils import format_papers_with_keys

logger = logging.getLogger(__name__)


async def write_intro_methodology_node(state: SynthesisState) -> dict[str, Any]:
    """Write introduction and methodology sections."""
    input_data = state.get("input", {})
    paper_summaries = state.get("paper_summaries", {})
    clusters = state.get("clusters", [])
    quality_settings = state.get("quality_settings", {})

    topic = input_data.get("topic", "Unknown topic")
    research_questions = input_data.get("research_questions", [])
    date_range = input_data.get("date_range")

    themes_overview = "\n".join(
        f"- {c['label']}: {c['description'][:100]}"
        for c in clusters
    )

    years = [s.get("year", 0) for s in paper_summaries.values() if s.get("year")]
    if years:
        actual_range = f"{min(years)}-{max(years)}"
    elif date_range:
        actual_range = f"{date_range[0]}-{date_range[1]}"
    else:
        actual_range = "Not specified"

    intro_prompt = INTRODUCTION_USER_TEMPLATE.format(
        topic=topic,
        research_questions="\n".join(f"- {q}" for q in research_questions),
        themes_overview=themes_overview,
        paper_count=len(paper_summaries),
        date_range=actual_range,
    )

    llm = get_llm(tier=ModelTier.SONNET, max_tokens=4096)

    intro_response = await invoke_with_cache(
        llm,
        system_prompt=INTRODUCTION_SYSTEM_PROMPT,
        user_prompt=intro_prompt,
    )

    introduction = intro_response.content if isinstance(intro_response.content, str) else intro_response.content[0].get("text", "")

    total_papers = len(paper_summaries)

    method_prompt = METHODOLOGY_USER_TEMPLATE.format(
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
        system_prompt=METHODOLOGY_SYSTEM_PROMPT,
        user_prompt=method_prompt,
    )

    methodology = method_response.content if isinstance(method_response.content, str) else method_response.content[0].get("text", "")

    logger.info("Completed introduction and methodology sections")

    return {
        "introduction_draft": introduction,
        "methodology_draft": methodology,
    }


async def write_thematic_sections_node(state: SynthesisState) -> dict[str, Any]:
    """Write a section for each thematic cluster (parallel)."""
    clusters = state.get("clusters", [])
    cluster_analyses = state.get("cluster_analyses", [])
    paper_summaries = state.get("paper_summaries", {})
    zotero_keys = state.get("zotero_keys", {})

    if not clusters:
        logger.warning("No clusters to write sections for")
        return {"thematic_section_drafts": {}}

    analysis_lookup = {a["cluster_id"]: a for a in cluster_analyses}

    async def write_single_section(cluster):
        """Write a single thematic section."""
        analysis = analysis_lookup.get(cluster["cluster_id"], {})

        papers_text = format_papers_with_keys(
            cluster["paper_dois"],
            paper_summaries,
            zotero_keys,
        )

        user_prompt = THEMATIC_SECTION_USER_TEMPLATE.format(
            theme_name=cluster["label"],
            theme_description=cluster["description"],
            sub_themes=", ".join(cluster.get("sub_themes", [])) or "None identified",
            key_debates="\n".join(f"- {d}" for d in cluster.get("conflicts", [])) or "None identified",
            outstanding_questions="\n".join(f"- {q}" for q in cluster.get("gaps", [])) or "None identified",
            papers_with_keys=papers_text,
            narrative_summary=analysis.get("narrative_summary", "No analysis available"),
        )

        try:
            llm = get_llm(tier=ModelTier.SONNET, max_tokens=6000)

            response = await invoke_with_cache(
                llm,
                system_prompt=THEMATIC_SECTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            section_text = response.content if isinstance(response.content, str) else response.content[0].get("text", "")

            return (cluster["label"], section_text)

        except Exception as e:
            logger.error(f"Failed to write section for {cluster['label']}: {e}")
            return (cluster["label"], f"[Section generation failed: {e}]")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SECTIONS)

    async def write_with_limit(cluster):
        async with semaphore:
            return await write_single_section(cluster)

    tasks = [write_with_limit(c) for c in clusters]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    section_drafts = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Section writing task failed: {result}")
            continue
        label, text = result
        section_drafts[label] = text

    logger.info(f"Completed {len(section_drafts)} thematic sections")

    return {"thematic_section_drafts": section_drafts}


async def write_discussion_conclusions_node(state: SynthesisState) -> dict[str, Any]:
    """Write discussion and conclusions sections."""
    input_data = state.get("input", {})
    clusters = state.get("clusters", [])
    cluster_analyses = state.get("cluster_analyses", [])

    research_questions = input_data.get("research_questions", [])

    themes_summary = "\n".join(
        f"- {c['label']}: {c['description'][:150]}"
        for c in clusters
    )

    cross_cutting = []
    gaps = []
    for analysis in cluster_analyses:
        for question in analysis.get("outstanding_questions", []):
            gaps.append(question)

    for cluster in clusters:
        for gap in cluster.get("gaps", []):
            gaps.append(gap)

    discussion_prompt = DISCUSSION_USER_TEMPLATE.format(
        research_questions="\n".join(f"- {q}" for q in research_questions),
        themes_summary=themes_summary,
        cross_cutting_findings="See thematic sections for detailed findings.",
        research_gaps="\n".join(f"- {g}" for g in gaps[:10]) or "None explicitly identified",
    )

    llm = get_llm(tier=ModelTier.SONNET, max_tokens=4096)

    discussion_response = await invoke_with_cache(
        llm,
        system_prompt=DISCUSSION_SYSTEM_PROMPT,
        user_prompt=discussion_prompt,
    )

    discussion = discussion_response.content if isinstance(discussion_response.content, str) else discussion_response.content[0].get("text", "")

    findings_summary = "\n".join(
        f"Q{i+1}: {q}\n   Finding: Based on {len(clusters)} themes covering {sum(len(c['paper_dois']) for c in clusters)} papers"
        for i, q in enumerate(research_questions)
    )

    conclusions_prompt = CONCLUSIONS_USER_TEMPLATE.format(
        research_questions="\n".join(f"- {q}" for q in research_questions),
        findings_per_question=findings_summary,
        main_contributions=f"Systematic review of {sum(len(c['paper_dois']) for c in clusters)} papers organized into {len(clusters)} themes",
    )

    conclusions_response = await invoke_with_cache(
        llm,
        system_prompt=CONCLUSIONS_SYSTEM_PROMPT,
        user_prompt=conclusions_prompt,
    )

    conclusions = conclusions_response.content if isinstance(conclusions_response.content, str) else conclusions_response.content[0].get("text", "")

    logger.info("Completed discussion and conclusions sections")

    return {
        "discussion_draft": discussion,
        "conclusions_draft": conclusions,
    }
