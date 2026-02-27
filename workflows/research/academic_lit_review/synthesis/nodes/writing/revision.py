"""Thematic section writing nodes.

Routes through unified invoke() for automatic broker routing and cost optimization.
"""

import logging
from typing import Any

from core.llm_broker import BatchPolicy
from workflows.shared.language import get_translated_prompt
from workflows.shared.llm_utils import ModelTier, invoke_batch, InvokeConfig
from workflows.shared.llm_utils.response_parsing import extract_response_content

from ...citation_utils import format_papers_with_keys
from ...types import SynthesisState
from .prompts import (
    DEFAULT_TARGET_WORDS,
    THEMATIC_SECTION_USER_TEMPLATE,
    get_thematic_section_system_prompt,
)

logger = logging.getLogger(__name__)


async def write_thematic_sections_node(state: SynthesisState) -> dict[str, Any]:
    """Write a section for each thematic cluster.

    Uses invoke_batch() for efficient batched LLM calls with automatic
    broker routing and cost optimization.
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

    cluster_labels = []

    logger.info(f"Submitting {len(clusters)} thematic sections")

    # Use invoke_batch for efficient batching
    async with invoke_batch() as batch:
        for cluster in clusters:
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
                key_debates="\n".join(f"- {d}" for d in cluster.get("conflicts", [])) or "None identified",
                outstanding_questions="\n".join(f"- {q}" for q in cluster.get("gaps", [])) or "None identified",
                papers_with_keys=papers_text,
                narrative_summary=analysis.get("narrative_summary", "No analysis available"),
            )

            batch.add(
                tier=ModelTier.SONNET,
                system=thematic_system,
                user=user_prompt,
                config=InvokeConfig(
                    batch_policy=BatchPolicy.PREFER_BALANCE,
                    max_tokens=6000,
                ),
            )
            cluster_labels.append(cluster["label"])

    # Collect results
    batch_results = await batch.results()
    section_drafts = {}
    for i, label in enumerate(cluster_labels):
        try:
            response = batch_results[i]
            section_drafts[label] = extract_response_content(response)
        except Exception as e:
            logger.error(f"Failed to write section for {label}: {e}")
            section_drafts[label] = f"[Section generation failed: {e}]"

    logger.info(f"Completed {len(section_drafts)} thematic sections")

    return {"thematic_section_drafts": section_drafts}
