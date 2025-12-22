"""
Translate final report node.

Translates the final research report to a target language using OPUS
for high-quality, citation-preserving translation.
"""

import logging
from typing import Any

from workflows.research.state import DeepResearchState
from workflows.shared.llm_utils import ModelTier, get_llm
from workflows.research.config.languages import LANGUAGE_NAMES

logger = logging.getLogger(__name__)


async def translate_report(state: DeepResearchState) -> dict[str, Any]:
    """Translate the final report to the target language.

    Uses OPUS for high-quality translation that:
    - Preserves exact meaning and nuance
    - Maintains all citations in original format
    - Keeps document structure intact
    - Optionally preserves quotes in original language

    Returns:
        - translated_report: Translated content
        - current_status: updated status
    """
    translation_config = state.get("translation_config")

    # Skip if translation not enabled
    if not translation_config or not translation_config.get("enabled"):
        return {}

    final_report = state.get("final_report")
    if not final_report:
        logger.warning("No final report available for translation")
        return {}

    target_lang_code = translation_config.get("target_language", "en")
    source_lang_code = state.get("primary_language", "en")
    preserve_quotes = translation_config.get("preserve_quotes", False)
    preserve_citations = translation_config.get("preserve_citations", True)

    # Get language names
    target_lang = LANGUAGE_NAMES.get(target_lang_code, target_lang_code)
    source_lang = LANGUAGE_NAMES.get(source_lang_code, source_lang_code)

    # Build translation prompt
    system_prompt = f"""You are a professional translator specializing in academic and research content.

Translate the research report from {source_lang} to {target_lang} while:
- Preserving EXACT meaning and nuance
- Maintaining formal, academic tone
- Keeping document structure (headers, sections, lists) unchanged
- {"Keeping all citations in EXACT original format: [1], [@KEY], (Author, Year), etc." if preserve_citations else "Translating citation metadata if present"}
- {"Preserving direct quotes in original language, adding [translated: ...] after each quote" if preserve_quotes else "Translating quotes naturally while maintaining attribution"}

Your translation must be publication-quality and read naturally in {target_lang}."""

    user_prompt = f"Translate this research report to {target_lang}:\n\n{final_report}"

    llm = get_llm(ModelTier.OPUS, max_tokens=16384)  # OPUS for quality, high token limit for long reports

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        # Handle response content
        if isinstance(response.content, str):
            translated = response.content.strip()
        elif isinstance(response.content, list) and len(response.content) > 0:
            first_block = response.content[0]
            if isinstance(first_block, dict):
                translated = first_block.get("text", "").strip()
            elif hasattr(first_block, "text"):
                translated = first_block.text.strip()
            else:
                translated = str(first_block).strip()
        else:
            translated = str(response.content).strip()

        logger.info(f"Translated report from {source_lang} to {target_lang}: {len(translated)} chars")

        return {
            "translated_report": translated,
            "current_status": "translation_complete",
        }

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return {
            "errors": [{"node": "translate_report", "error": str(e)}],
            "current_status": "translation_failed",
        }
