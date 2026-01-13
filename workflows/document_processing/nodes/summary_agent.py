"""
Summary generation node for document processing workflow.

Uses prompt caching for 90% cost reduction when processing multiple documents.
Generates dual summaries (original language + English) for non-English documents.
"""

import logging
from typing import Any

from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.language import LANGUAGE_NAMES
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from workflows.shared.llm_utils.response_parsing import extract_response_content
from workflows.shared.retry_utils import with_retry
from workflows.shared.text_utils import get_first_n_pages, get_last_n_pages

logger = logging.getLogger(__name__)

# System prompt for summarization (cached)
SUMMARY_SYSTEM = """You are a skilled summarizer. Create concise summaries that capture the essential information.

Guidelines:
- Focus on the main thesis, key arguments, and conclusions
- Preserve critical details and nuance
- Write in clear, professional prose"""

# System prompt for translation (kept simple for summaries)
TRANSLATION_SYSTEM = """You are a skilled translator. Translate the following text accurately to English while:
- Preserving the meaning and nuance
- Maintaining academic/professional tone
- Keeping technical terms appropriately translated or retained

Output ONLY the English translation, no explanations or preamble."""


async def generate_summary(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Generate summary of the document.

    For very long docs (>50k chars): uses first+last 10 pages.
    For non-English docs: generates original language summary + English translation.
    Returns short_summary, short_summary_original, and short_summary_english.
    """
    try:
        processing_result = state.get("processing_result")
        if not processing_result:
            logger.error("No processing_result in state")
            return {
                "errors": [{"node": "summary_agent", "error": "No processing result"}],
            }

        markdown = processing_result["markdown"]
        original_language = state.get("original_language", "en")

        # For very long documents, use first and last pages
        if len(markdown) > 50000:
            logger.info("Document is long, using first and last 10 pages for summary")
            first_pages = get_first_n_pages(markdown, 10)
            last_pages = get_last_n_pages(markdown, 10)
            content = f"{first_pages}\n\n[... middle section omitted ...]\n\n{last_pages}"
        else:
            content = markdown

        # Build user prompt with language instruction if non-English
        lang_instruction = ""
        if original_language != "en":
            lang_name = LANGUAGE_NAMES.get(original_language, original_language)
            lang_instruction = f" Write the summary in {lang_name}."

        user_prompt = f"""Summarize the following text in approximately 100 words.{lang_instruction}

Focus on the main thesis, key arguments, and conclusions. Highlight what makes this work significant.

Text:
{content}"""

        # Generate summary via LLM with prompt caching
        llm = get_llm(tier=ModelTier.SONNET)

        async def _summarize():
            response = await invoke_with_cache(
                llm,
                system_prompt=SUMMARY_SYSTEM,
                user_prompt=user_prompt,
            )
            return extract_response_content(response)

        original_summary = (await with_retry(_summarize)).strip()

        logger.info(
            f"Generated original summary ({len(original_summary.split())} words, "
            f"lang={original_language})"
        )

        # Build result with backward compatibility
        result = {
            "short_summary": original_summary,  # Backward compatibility
            "short_summary_original": original_summary,
        }

        # If non-English, also generate English translation
        if original_language != "en":
            english_summary = await _translate_to_english(original_summary)
            result["short_summary_english"] = english_summary.strip()
            logger.info(f"Generated English translation ({len(english_summary.split())} words)")

        return result

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return {
            "errors": [{"node": "summary_agent", "error": str(e)}],
        }


async def _translate_to_english(text: str) -> str:
    """Translate text to English using Sonnet."""
    llm = get_llm(tier=ModelTier.SONNET)

    async def _invoke():
        response = await invoke_with_cache(
            llm,
            system_prompt=TRANSLATION_SYSTEM,
            user_prompt=f"Translate this text to English:\n\n{text}",
        )
        return extract_response_content(response)

    return await with_retry(_invoke)
