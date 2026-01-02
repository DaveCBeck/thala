"""
Recommendation generation nodes using Opus.

Three parallel nodes that each call Opus to generate book recommendations
for their specific category.
"""

import json
import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm
from workflows.research.subgraphs.book_finding.state import (
    BookRecommendation,
    BookFindingQualitySettings,
    BOOK_QUALITY_PRESETS,
)
from workflows.research.subgraphs.book_finding.prompts import get_recommendation_prompts

logger = logging.getLogger(__name__)


def _format_brief_section(brief: str | None) -> str:
    """Format optional brief for inclusion in prompt."""
    if brief:
        return f"\nAdditional context: {brief}\n"
    return ""


async def _generate_recommendations(
    theme: str,
    brief: str | None,
    category: str,
    system_prompt: str,
    user_template: str,
    quality_settings: BookFindingQualitySettings,
) -> list[BookRecommendation]:
    """Generate book recommendations using configured model.

    Args:
        theme: The theme to explore
        brief: Optional additional context
        category: Category name for the recommendations
        system_prompt: System prompt for this category
        user_template: User prompt template
        quality_settings: Quality configuration for model and token limits

    Returns:
        List of BookRecommendation objects
    """
    # Select model based on quality settings
    model_tier = ModelTier.OPUS if quality_settings["use_opus_for_recommendations"] else ModelTier.SONNET
    max_tokens = quality_settings["recommendation_max_tokens"]
    max_recs = quality_settings["recommendations_per_category"]

    llm = get_llm(model_tier, max_tokens=max_tokens)

    brief_section = _format_brief_section(brief)
    user_prompt = user_template.format(theme=theme, brief_section=brief_section)

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        content = response.content if isinstance(response.content, str) else str(response.content)

        # Extract JSON from response (handle code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        recommendations_raw = json.loads(content.strip())

        recommendations = [
            BookRecommendation(
                title=r["title"],
                author=r.get("author"),
                explanation=r["explanation"],
                category=category,
            )
            for r in recommendations_raw[:max_recs]
        ]

        logger.info(f"Generated {len(recommendations)} {category} recommendations for theme: {theme[:50]}...")
        return recommendations

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {category} recommendations JSON: {e}")
        logger.debug(f"Raw content: {content[:500]}...")
        return []
    except Exception as e:
        logger.error(f"Failed to generate {category} recommendations: {e}")
        return []


async def generate_analogous_recommendations(state: dict) -> dict[str, Any]:
    """Generate analogous domain book recommendations.

    Finds books that explore similar themes in different domains,
    providing unexpected perspectives through cross-disciplinary connections.
    """
    # Handle both direct state (from Send) and nested input
    theme = state.get("theme") or state.get("input", {}).get("theme", "")
    brief = state.get("brief") or state.get("input", {}).get("brief")
    quality_settings = state.get("quality_settings") or BOOK_QUALITY_PRESETS["standard"]
    language_config = state.get("language_config")

    system_prompt, user_template = await get_recommendation_prompts("analogous", language_config)

    recs = await _generate_recommendations(
        theme=theme,
        brief=brief,
        category="analogous",
        system_prompt=system_prompt,
        user_template=user_template,
        quality_settings=quality_settings,
    )

    return {"analogous_recommendations": recs}


async def generate_inspiring_recommendations(state: dict) -> dict[str, Any]:
    """Generate inspiring action book recommendations.

    Finds fiction and nonfiction that inspires action, change, or
    transformation related to the theme.
    """
    theme = state.get("theme") or state.get("input", {}).get("theme", "")
    brief = state.get("brief") or state.get("input", {}).get("brief")
    quality_settings = state.get("quality_settings") or BOOK_QUALITY_PRESETS["standard"]
    language_config = state.get("language_config")

    system_prompt, user_template = await get_recommendation_prompts("inspiring", language_config)

    recs = await _generate_recommendations(
        theme=theme,
        brief=brief,
        category="inspiring",
        system_prompt=system_prompt,
        user_template=user_template,
        quality_settings=quality_settings,
    )

    return {"inspiring_recommendations": recs}


async def generate_expressive_recommendations(state: dict) -> dict[str, Any]:
    """Generate expressive fiction book recommendations.

    Finds fiction that captures the lived experience or potential
    of the theme - what it feels like or could become.
    """
    theme = state.get("theme") or state.get("input", {}).get("theme", "")
    brief = state.get("brief") or state.get("input", {}).get("brief")
    quality_settings = state.get("quality_settings") or BOOK_QUALITY_PRESETS["standard"]
    language_config = state.get("language_config")

    system_prompt, user_template = await get_recommendation_prompts("expressive", language_config)

    recs = await _generate_recommendations(
        theme=theme,
        brief=brief,
        category="expressive",
        system_prompt=system_prompt,
        user_template=user_template,
        quality_settings=quality_settings,
    )

    return {"expressive_recommendations": recs}
