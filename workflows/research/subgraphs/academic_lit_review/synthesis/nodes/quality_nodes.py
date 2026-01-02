"""Quality verification nodes for synthesis subgraph."""

import json
import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from workflows.shared.language import get_translated_prompt
from ..types import SynthesisState
from ..prompts import QUALITY_CHECK_SYSTEM_PROMPT
from ..citation_utils import calculate_quality_metrics

logger = logging.getLogger(__name__)


async def verify_quality_node(state: SynthesisState) -> dict[str, Any]:
    """Verify quality of the final review."""
    final_review = state.get("final_review", "")
    paper_summaries = state.get("paper_summaries", {})
    zotero_keys = state.get("zotero_keys", {})
    quality_settings = state.get("quality_settings", {})
    language_config = state.get("language_config")

    metrics = calculate_quality_metrics(final_review, paper_summaries, zotero_keys)

    target_words = quality_settings.get("target_word_count", 10000)
    quality_passed = (
        metrics["corpus_coverage"] >= 0.4 and
        metrics["total_words"] >= target_words * 0.7 and
        len(metrics["issues"]) <= 2
    )

    try:
        llm = get_llm(tier=ModelTier.HAIKU, max_tokens=2000)

        sample = final_review[:5000]

        # Translate prompt if needed
        quality_system = QUALITY_CHECK_SYSTEM_PROMPT
        if language_config and language_config["code"] != "en":
            quality_system = await get_translated_prompt(
                QUALITY_CHECK_SYSTEM_PROMPT,
                language_code=language_config["code"],
                language_name=language_config["name"],
                prompt_name="lit_review_quality_system",
            )

        response = await invoke_with_cache(
            llm,
            system_prompt=quality_system,
            user_prompt=f"Review this literature review sample for quality:\n\n{sample}",
        )

        content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")

        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        quality_result = json.loads(content)
        additional_issues = quality_result.get("issues", [])

        if additional_issues:
            metrics["issues"].extend(additional_issues[:5])

        if quality_result.get("overall_quality") == "needs_revision":
            quality_passed = False

    except Exception as e:
        logger.warning(f"LLM quality check failed: {e}")

    logger.info(
        f"Quality check: {metrics['total_words']} words, "
        f"{metrics['unique_papers_cited']} papers cited ({metrics['corpus_coverage']:.0%} coverage), "
        f"passed={quality_passed}"
    )

    return {
        "quality_metrics": metrics,
        "quality_passed": quality_passed,
    }
