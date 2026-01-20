"""Integration nodes for synthesis subgraph.

Programmatically assembles document sections and uses LLM only for abstract generation.
This avoids token limit issues from having the LLM re-output the entire document.
"""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from workflows.shared.language import get_translated_prompt
from ..types import SynthesisState
from ..prompts import (
    get_abstract_system_prompt,
    ABSTRACT_USER_TEMPLATE,
    SECTION_PROPORTIONS,
    DEFAULT_TARGET_WORDS,
)
from ..citation_utils import extract_citations_from_text

logger = logging.getLogger(__name__)


def _assemble_document(
    topic: str,
    introduction: str,
    methodology: str,
    thematic_sections: dict[str, str],
    discussion: str,
    conclusions: str,
    clusters: list[dict],
    abstract: str = "",
) -> str:
    """Programmatically assemble the literature review document.

    Combines all sections with proper markdown headers. No LLM needed.
    """
    parts = [f"# Literature Review: {topic}\n"]

    if abstract:
        parts.append(f"## Abstract\n\n{abstract}\n")

    parts.append(f"## 1. Introduction\n\n{introduction}\n")
    parts.append(f"## 2. Methodology\n\n{methodology}\n")

    # Add thematic sections in cluster order
    cluster_order = [c["label"] for c in clusters]
    for i, label in enumerate(cluster_order):
        section_num = i + 3
        section_text = thematic_sections.get(
            label, f"[Section for {label} not available]"
        )
        parts.append(f"## {section_num}. {label}\n\n{section_text}\n")

    # Discussion and conclusions
    discussion_num = len(cluster_order) + 3
    conclusions_num = discussion_num + 1

    parts.append(f"## {discussion_num}. Discussion\n\n{discussion}\n")
    parts.append(f"## {conclusions_num}. Conclusions\n\n{conclusions}\n")

    return "\n".join(parts)


async def integrate_sections_node(state: SynthesisState) -> dict[str, Any]:
    """Integrate all sections into a cohesive document.

    Uses programmatic assembly for the document structure and LLM only
    for generating the abstract. This avoids token limit issues and
    ensures no content is lost during integration.
    """
    input_data = state.get("input", {})
    introduction = state.get("introduction_draft", "")
    methodology = state.get("methodology_draft", "")
    thematic_sections = state.get("thematic_section_drafts", {})
    discussion = state.get("discussion_draft", "")
    conclusions = state.get("conclusions_draft", "")
    clusters = state.get("clusters", [])
    language_config = state.get("language_config")
    quality_settings = state.get("quality_settings", {})

    topic = input_data.get("topic", "Literature Review")
    target_words = quality_settings.get("target_word_count", DEFAULT_TARGET_WORDS)

    # Step 1: Generate abstract using LLM (only ~250 words output)
    abstract_target = int(target_words * SECTION_PROPORTIONS["abstract"])
    abstract_system = get_abstract_system_prompt(abstract_target)
    abstract_user_template = ABSTRACT_USER_TEMPLATE

    if language_config and language_config["code"] != "en":
        abstract_system = await get_translated_prompt(
            abstract_system,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_abstract_system",
        )
        abstract_user_template = await get_translated_prompt(
            ABSTRACT_USER_TEMPLATE,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_abstract_user",
        )

    abstract_prompt = abstract_user_template.format(
        topic=topic,
        introduction=introduction,
        conclusions=conclusions,
    )

    llm = get_llm(tier=ModelTier.SONNET, max_tokens=1000)

    response = await invoke_with_cache(
        llm,
        system_prompt=abstract_system,
        user_prompt=abstract_prompt,
        cache_ttl="1h",
    )

    abstract = (
        response.content
        if isinstance(response.content, str)
        else response.content[0].get("text", "")
    )

    logger.info(f"Generated abstract: {len(abstract.split())} words")

    # Step 2: Assemble final document with abstract
    integrated = _assemble_document(
        topic=topic,
        introduction=introduction,
        methodology=methodology,
        thematic_sections=thematic_sections,
        discussion=discussion,
        conclusions=conclusions,
        clusters=clusters,
        abstract=abstract,
    )

    # Count citations for logging
    output_citations = extract_citations_from_text(integrated)

    logger.info(
        f"Assembled review: {len(integrated.split())} words, "
        f"{len(output_citations)} citations preserved"
    )

    return {"integrated_review": integrated}
