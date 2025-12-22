"""
Analyze languages node.

Analyzes which languages would provide unique insights for composite multi-lingual research.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from workflows.research.state import DeepResearchState, LanguageConfig
from workflows.research.config.languages import get_language_config, LANGUAGE_NAMES
from workflows.shared.llm_utils import get_llm, ModelTier

logger = logging.getLogger(__name__)


class LanguageRecommendation(BaseModel):
    """Recommendation for a specific language to research."""

    language_code: str = Field(description="ISO 639-1 language code (e.g., 'es', 'zh')")
    rationale: str = Field(description="Why this language adds unique value")
    expected_unique_insights: list[str] = Field(
        description="Specific insights expected from this language"
    )
    priority: int = Field(description="Priority ranking (1=highest)", ge=1)


class LanguageAnalysisResult(BaseModel):
    """Result of language analysis for multi-lingual research."""

    recommendations: list[LanguageRecommendation] = Field(
        description="Recommended languages to research, ordered by priority",
        max_length=5,
    )
    reasoning: str = Field(description="Overall reasoning for language selection")


ANALYZE_LANGUAGES_SYSTEM = """You are an expert in identifying language-specific research opportunities.

Your task is to analyze a research topic and recommend which languages would provide UNIQUE, valuable insights that aren't readily available in English sources.

CRITICAL GUIDELINES:
- Only recommend languages that offer genuinely unique perspectives, sources, or expertise
- Do NOT recommend a language just because it's widely spoken globally
- Consider: academic conferences, specialized journals, regional expertise, cultural perspectives, government/institutional reports
- Expected overlap vs. unique information is key - if English sources cover it well, don't recommend other languages
- Maximum 3-4 languages unless the topic is exceptionally global
- Priority 1 = most valuable, 2 = valuable, 3 = marginal value

ANALYSIS FRAMEWORK:
1. Where is this topic primarily discussed? (academic venues, news, industry forums)
2. Which countries/cultures have:
   - Strong research institutions or expertise on this topic?
   - Unique cultural/historical perspectives?
   - Language-specific sources (journals, databases, experts)?
3. What percentage of information is unique vs. already available in English?
4. Are there language barriers limiting English coverage?

Available languages: {languages}

Respond with structured analysis."""

ANALYZE_LANGUAGES_HUMAN = """Research Topic: {topic}

Research Objectives:
{objectives}

Research Scope:
{scope}

Key Questions:
{questions}

Analyze which languages would provide unique insights for this research."""


async def analyze_languages(state: DeepResearchState) -> dict[str, Any]:
    """Analyze which languages would provide unique insights for this topic.

    Only runs in composite multi-lingual mode. If target_languages are already
    specified, uses those instead of analyzing.

    Returns:
        - active_languages: list of language codes
        - language_configs: dict mapping codes to LanguageConfig
        - current_status: updated status
    """
    input_data = state["input"]

    # Only run if multi_lingual is enabled
    if not input_data.get("multi_lingual"):
        logger.debug("Multi-lingual mode not enabled, skipping language analysis")
        return {
            "current_status": state.get("current_status", "searching_memory"),
        }

    # If target_languages already specified, use those
    target_languages = input_data.get("target_languages")
    if target_languages:
        logger.info(f"Using pre-specified target languages: {target_languages}")

        # Build language configs
        language_configs = {}
        for code in target_languages:
            config = get_language_config(code)
            if config:
                language_configs[code] = config
            else:
                logger.warning(f"Unsupported language code: {code}")

        if not language_configs:
            logger.error("No valid language configs generated from target_languages")
            return {
                "errors": [{
                    "node": "analyze_languages",
                    "error": f"No valid language configs for: {target_languages}"
                }],
                "current_status": state.get("current_status", "searching_memory"),
            }

        return {
            "active_languages": list(language_configs.keys()),
            "language_configs": language_configs,
            "current_status": "languages_analyzed",
        }

    # Otherwise, use Opus to analyze and recommend languages
    brief = state.get("research_brief")
    if not brief:
        logger.warning("No research brief available for language analysis")
        return {
            "errors": [{
                "node": "analyze_languages",
                "error": "No research brief available"
            }],
            "current_status": state.get("current_status", "searching_memory"),
        }

    # Format prompt
    available_languages = ", ".join([f"{code} ({name})" for code, name in LANGUAGE_NAMES.items()])
    system_prompt = ANALYZE_LANGUAGES_SYSTEM.format(languages=available_languages)

    objectives = "\n".join([f"- {obj}" for obj in brief.get("objectives", [])])
    questions = "\n".join([f"- {q}" for q in brief.get("key_questions", [])])

    human_prompt = ANALYZE_LANGUAGES_HUMAN.format(
        topic=brief["topic"],
        objectives=objectives or "None specified",
        scope=brief.get("scope", "General research"),
        questions=questions or "None specified",
    )

    # Use Opus for complex reasoning
    llm = get_llm(ModelTier.OPUS)

    try:
        # Use structured output
        structured_llm = llm.with_structured_output(LanguageAnalysisResult)

        response = await structured_llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ])

        if not response.recommendations:
            logger.info("No languages recommended by analysis - continuing with English only")
            return {
                "active_languages": [],
                "language_configs": {},
                "current_status": "languages_analyzed",
            }

        # Build language configs from recommendations
        language_configs = {}
        active_languages = []

        for rec in response.recommendations:
            config = get_language_config(rec.language_code)
            if config:
                language_configs[rec.language_code] = config
                active_languages.append(rec.language_code)
                logger.info(
                    f"Language recommended: {rec.language_code} ({config['name']}) - "
                    f"Priority {rec.priority} - {rec.rationale}"
                )
            else:
                logger.warning(f"Recommended language not supported: {rec.language_code}")

        if not language_configs:
            logger.warning("No supported languages in recommendations - continuing with English only")
            return {
                "active_languages": [],
                "language_configs": {},
                "current_status": "languages_analyzed",
            }

        logger.info(
            f"Language analysis complete: {len(active_languages)} languages - {active_languages}"
        )

        return {
            "active_languages": active_languages,
            "language_configs": language_configs,
            "current_status": "languages_analyzed",
        }

    except Exception as e:
        logger.error(f"Language analysis failed: {e}")

        return {
            "errors": [{
                "node": "analyze_languages",
                "error": str(e)
            }],
            "active_languages": [],
            "language_configs": {},
            "current_status": "languages_analyzed",
        }
