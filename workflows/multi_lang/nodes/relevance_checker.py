"""Relevance checking node for multi-lingual research workflow."""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from langchain_tools.firecrawl import web_search
from workflows.multi_lang.prompts.relevance import (
    RELEVANCE_CHECK_SYSTEM,
    RELEVANCE_CHECK_USER,
)
from workflows.multi_lang.state import LanguageRelevanceCheck, MultiLangState
from workflows.shared.language.query_translator import translate_query
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


class RelevanceDecision(BaseModel):
    """Structured output for Haiku relevance check."""

    has_meaningful_discussion: bool = Field(
        description="Whether meaningful discussion exists in this language"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence 0-1")
    reasoning: str = Field(description="Brief explanation (1-2 sentences)")
    suggested_depth: Literal["skip", "quick", "standard", "comprehensive"] = Field(
        description="Recommended search depth if proceeding"
    )


async def _quick_web_search(query: str, language_config: dict) -> list[dict]:
    """Run a quick web search in the target language.

    Returns list of {title, url, description} dicts (limit 5 results).
    """
    code = language_config["code"]
    locale = language_config.get("locale")
    preferred_domains = language_config.get("preferred_domains")

    try:
        # Use web_search tool with language hints
        search_result = await web_search(
            query=query, limit=5, locale=locale, preferred_domains=preferred_domains
        )

        # Extract results
        results = search_result.get("results", [])

        # Convert to simple dict format
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("description", ""),
            }
            for r in results
        ]

    except Exception as e:
        logger.warning(f"Quick web search failed for {code}: {e}")
        return []


async def _check_language_relevance(
    topic: str,
    research_questions: list[str],
    language_code: str,
    language_config: dict,
) -> LanguageRelevanceCheck:
    """Check if meaningful discussion exists for topic in target language."""
    language_name = language_config["name"]

    # Translate query to target language
    translated_query = await translate_query(
        topic, target_language_code=language_code, target_language_name=language_name
    )

    # Run quick web search
    search_results = await _quick_web_search(translated_query, language_config)

    # Format search results for prompt
    if search_results:
        results_text = "\n".join(
            f"- {r['title']}\n  URL: {r['url']}\n  {r['description'][:200] if r['description'] else 'No description'}"
            for r in search_results[:5]
        )
    else:
        results_text = "(No search results found)"

    # Format research questions
    questions_text = "\n".join(f"- {q}" for q in research_questions) if research_questions else "(No specific questions provided)"

    # Build prompts
    system_prompt = RELEVANCE_CHECK_SYSTEM.format(language_name=language_name)
    user_prompt = RELEVANCE_CHECK_USER.format(
        topic=topic,
        research_questions=questions_text,
        language_name=language_name,
        quick_search_results=results_text,
    )

    try:
        decision: RelevanceDecision = await get_structured_output(
            output_schema=RelevanceDecision,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tier=ModelTier.HAIKU,
            max_tokens=512,
        )

        return {
            "language_code": language_code,
            "has_meaningful_discussion": decision.has_meaningful_discussion,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "suggested_depth": decision.suggested_depth,
        }

    except Exception as e:
        logger.error(f"Relevance check failed for {language_code}: {e}")
        # Default to skip on error
        return {
            "language_code": language_code,
            "has_meaningful_discussion": False,
            "confidence": 0.5,
            "reasoning": f"Error during relevance check: {str(e)[:100]}",
            "suggested_depth": "skip",
        }


async def check_relevance_batch(state: MultiLangState) -> dict:
    """
    Run quick searches and Haiku relevance checks for all target languages.

    For each language (except English which always passes):
    1. Translate query to target language
    2. Run quick web search
    3. Ask Haiku with structured output if meaningful discussion exists

    Process SEQUENTIALLY (hardware constraint mentioned by user).

    Returns:
        relevance_checks: list[LanguageRelevanceCheck]
        current_phase: "relevance_checking"
        current_status: "Checked N languages, M have content"
    """
    target_languages = state["target_languages"]
    language_configs = state["language_configs"]
    topic = state["input"]["topic"]
    research_questions = state["input"].get("research_questions") or []

    relevance_checks = []

    for lang_code in target_languages:
        language_config = language_configs.get(lang_code)
        if not language_config:
            logger.warning(f"No config for {lang_code}, skipping")
            continue

        # English always passes
        if lang_code == "en":
            relevance_checks.append(
                {
                    "language_code": "en",
                    "has_meaningful_discussion": True,
                    "confidence": 1.0,
                    "reasoning": "Baseline language",
                    "suggested_depth": "standard",
                }
            )
            logger.info("English: Baseline language (auto-pass)")
            continue

        # Check relevance for other languages
        logger.info(f"Checking relevance for {language_config['name']}...")
        check = await _check_language_relevance(
            topic, research_questions, lang_code, language_config
        )
        relevance_checks.append(check)

        decision_text = (
            f"{'✓' if check['has_meaningful_discussion'] else '✗'} "
            f"(confidence: {check['confidence']:.2f}, depth: {check['suggested_depth']})"
        )
        logger.info(f"{language_config['name']}: {decision_text}")
        logger.debug(f"  Reasoning: {check['reasoning']}")

    # Count languages with content
    languages_with_content = sum(
        1 for check in relevance_checks if check["has_meaningful_discussion"]
    )

    return {
        "relevance_checks": relevance_checks,
        "current_phase": "relevance_checking",
        "current_status": f"Checked {len(relevance_checks)} languages, {languages_with_content} have content",
    }


async def filter_relevant_languages(state: MultiLangState) -> dict:
    """
    Filter to languages that have meaningful discussion.

    Uses the relevance_checks from previous node.
    Includes language if:
    - has_meaningful_discussion=True AND confidence >= 0.5
    - OR it's English (always included)

    Also updates quality settings based on suggested_depth from Haiku.

    Returns:
        languages_with_content: list[str]
        current_status: "Filtered to N languages with content"
    """
    relevance_checks = state.get("relevance_checks", [])
    quality_settings = state["input"]["quality_settings"]

    languages_with_content = []
    per_language_overrides = quality_settings.get("per_language_overrides", {}).copy()

    for check in relevance_checks:
        lang_code = check["language_code"]
        has_discussion = check["has_meaningful_discussion"]
        confidence = check["confidence"]
        suggested_depth = check["suggested_depth"]

        # Include if meaningful discussion with sufficient confidence, or if English
        if (has_discussion and confidence >= 0.5) or lang_code == "en":
            languages_with_content.append(lang_code)

            # Update quality settings based on suggested_depth
            if suggested_depth != "skip" and lang_code != "en":
                # Map suggested_depth to quality_tier (skip "skip" option)
                quality_tier = suggested_depth if suggested_depth != "skip" else "quick"

                per_language_overrides[lang_code] = {
                    "language_code": lang_code,
                    "quality_tier": quality_tier,
                }
                logger.debug(
                    f"Updated quality for {lang_code}: {quality_tier} (suggested by Haiku)"
                )

    # Update quality settings with new overrides
    updated_quality_settings = {
        "default_quality": quality_settings["default_quality"],
        "per_language_overrides": per_language_overrides,
    }

    # Update the input state with new quality settings
    updated_input = state["input"].copy()
    updated_input["quality_settings"] = updated_quality_settings

    language_names = ", ".join(
        state["language_configs"][code]["name"] for code in languages_with_content
    )

    return {
        "languages_with_content": languages_with_content,
        "input": updated_input,
        "current_status": f"Filtered to {len(languages_with_content)} languages with content: {language_names}",
    }
