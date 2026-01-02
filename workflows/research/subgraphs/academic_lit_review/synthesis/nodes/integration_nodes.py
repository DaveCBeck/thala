"""Integration nodes for synthesis subgraph."""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from workflows.shared.language import get_translated_prompt
from ..types import SynthesisState
from ..prompts import INTEGRATION_SYSTEM_PROMPT, INTEGRATION_USER_TEMPLATE

logger = logging.getLogger(__name__)


async def integrate_sections_node(state: SynthesisState) -> dict[str, Any]:
    """Integrate all sections into a cohesive document."""
    input_data = state.get("input", {})
    introduction = state.get("introduction_draft", "")
    methodology = state.get("methodology_draft", "")
    thematic_sections = state.get("thematic_section_drafts", {})
    discussion = state.get("discussion_draft", "")
    conclusions = state.get("conclusions_draft", "")
    clusters = state.get("clusters", [])
    language_config = state.get("language_config")

    topic = input_data.get("topic", "Literature Review")

    cluster_order = [c["label"] for c in clusters]
    ordered_sections = []

    for i, label in enumerate(cluster_order):
        section_text = thematic_sections.get(label, f"[Section for {label} not available]")
        ordered_sections.append(f"### {i + 3}. {label}\n\n{section_text}")

    thematic_text = "\n\n".join(ordered_sections)

    # Translate prompts if needed
    integration_system = INTEGRATION_SYSTEM_PROMPT
    integration_user_template = INTEGRATION_USER_TEMPLATE

    if language_config and language_config["code"] != "en":
        integration_system = await get_translated_prompt(
            INTEGRATION_SYSTEM_PROMPT,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_integration_system",
        )
        integration_user_template = await get_translated_prompt(
            INTEGRATION_USER_TEMPLATE,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_integration_user",
        )

    integration_prompt = integration_user_template.format(
        title=f"Literature Review: {topic}",
        introduction=introduction,
        methodology=methodology,
        thematic_sections=thematic_text,
        discussion=discussion,
        conclusions=conclusions,
    )

    llm = get_llm(tier=ModelTier.SONNET, max_tokens=16000)

    response = await invoke_with_cache(
        llm,
        system_prompt=integration_system,
        user_prompt=integration_prompt,
        cache_ttl="1h",
    )

    integrated = response.content if isinstance(response.content, str) else response.content[0].get("text", "")

    logger.info(f"Integrated review: {len(integrated.split())} words")

    return {"integrated_review": integrated}
