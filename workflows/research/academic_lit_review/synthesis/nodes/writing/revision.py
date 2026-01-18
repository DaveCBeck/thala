"""Thematic section writing nodes with batch processing support.

Uses Anthropic Batch API for 50% cost reduction when writing 5+ thematic sections.
"""

import asyncio
import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from workflows.shared.language import get_translated_prompt
from workflows.shared.batch_processor import BatchProcessor
from ...types import SynthesisState, MAX_CONCURRENT_SECTIONS
from ...citation_utils import format_papers_with_keys
from .prompts import (
    get_thematic_section_system_prompt,
    THEMATIC_SECTION_USER_TEMPLATE,
    DEFAULT_TARGET_WORDS,
)

logger = logging.getLogger(__name__)


async def write_thematic_sections_node(state: SynthesisState) -> dict[str, Any]:
    """Write a section for each thematic cluster.

    Uses Anthropic Batch API for 50% cost reduction when writing 5+ sections.
    """
    clusters = state.get("clusters", [])
    cluster_analyses = state.get("cluster_analyses", [])
    paper_summaries = state.get("paper_summaries", {})
    zotero_keys = state.get("zotero_keys", {})
    language_config = state.get("language_config")
    quality_settings = state.get("quality_settings", {})

    if not clusters:
        logger.warning("No clusters to write sections for")
        return {"thematic_section_drafts": {}}

    analysis_lookup = {a["cluster_id"]: a for a in cluster_analyses}

    target_words = quality_settings.get("target_word_count", DEFAULT_TARGET_WORDS)
    theme_count = len(clusters)
    thematic_system = get_thematic_section_system_prompt(target_words, theme_count)
    thematic_user_template = THEMATIC_SECTION_USER_TEMPLATE

    if language_config and language_config["code"] != "en":
        thematic_system = await get_translated_prompt(
            thematic_system,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_thematic_system",
        )
        thematic_user_template = await get_translated_prompt(
            THEMATIC_SECTION_USER_TEMPLATE,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_thematic_user",
        )

    if len(clusters) >= 5:
        return await _write_thematic_sections_batched(
            clusters,
            analysis_lookup,
            paper_summaries,
            zotero_keys,
            thematic_system,
            thematic_user_template,
        )

    async def write_single_section(cluster):
        """Write a single thematic section."""
        analysis = analysis_lookup.get(cluster["cluster_id"], {})

        papers_text = format_papers_with_keys(
            cluster["paper_dois"],
            paper_summaries,
            zotero_keys,
        )

        user_prompt = thematic_user_template.format(
            theme_name=cluster["label"],
            theme_description=cluster["description"],
            sub_themes=", ".join(cluster.get("sub_themes", [])) or "None identified",
            key_debates="\n".join(f"- {d}" for d in cluster.get("conflicts", []))
            or "None identified",
            outstanding_questions="\n".join(f"- {q}" for q in cluster.get("gaps", []))
            or "None identified",
            papers_with_keys=papers_text,
            narrative_summary=analysis.get(
                "narrative_summary", "No analysis available"
            ),
        )

        try:
            llm = get_llm(tier=ModelTier.SONNET, max_tokens=6000)

            response = await invoke_with_cache(
                llm,
                system_prompt=thematic_system,
                user_prompt=user_prompt,
            )

            section_text = (
                response.content
                if isinstance(response.content, str)
                else response.content[0].get("text", "")
            )

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


async def _write_thematic_sections_batched(
    clusters: list,
    analysis_lookup: dict,
    paper_summaries: dict,
    zotero_keys: dict,
    thematic_system: str,
    thematic_user_template: str,
) -> dict[str, Any]:
    """Write thematic sections using Anthropic Batch API for 50% cost reduction."""
    processor = BatchProcessor(poll_interval=30)

    cluster_labels = []
    for i, cluster in enumerate(clusters):
        analysis = analysis_lookup.get(cluster["cluster_id"], {})

        papers_text = format_papers_with_keys(
            cluster["paper_dois"],
            paper_summaries,
            zotero_keys,
        )

        user_prompt = thematic_user_template.format(
            theme_name=cluster["label"],
            theme_description=cluster["description"],
            sub_themes=", ".join(cluster.get("sub_themes", [])) or "None identified",
            key_debates="\n".join(f"- {d}" for d in cluster.get("conflicts", []))
            or "None identified",
            outstanding_questions="\n".join(f"- {q}" for q in cluster.get("gaps", []))
            or "None identified",
            papers_with_keys=papers_text,
            narrative_summary=analysis.get(
                "narrative_summary", "No analysis available"
            ),
        )

        processor.add_request(
            custom_id=f"section-{i}",
            prompt=user_prompt,
            model=ModelTier.SONNET,
            max_tokens=6000,
            system=thematic_system,
        )
        cluster_labels.append(cluster["label"])

    logger.info(f"Submitting batch of {len(clusters)} thematic sections")
    results = await processor.execute_batch()

    section_drafts = {}
    for i, label in enumerate(cluster_labels):
        result = results.get(f"section-{i}")
        if result and result.success:
            section_drafts[label] = result.content
        else:
            error_msg = result.error if result else "No result returned"
            logger.error(f"Failed to write section for {label}: {error_msg}")
            section_drafts[label] = f"[Section generation failed: {error_msg}]"

    logger.info(f"Completed {len(section_drafts)} thematic sections (batch)")

    return {"thematic_section_drafts": section_drafts}
