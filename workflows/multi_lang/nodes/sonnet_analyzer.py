"""Sonnet-powered cross-language analysis producing comparative documents."""

import logging
from datetime import datetime
from pydantic import BaseModel, Field

from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.multi_lang.prompts.analysis import (
    CROSS_ANALYSIS_SYSTEM,
    CROSS_ANALYSIS_USER,
)
from workflows.multi_lang.state import (
    MultiLangState,
    LanguageResult,
    SonnetCrossAnalysis,
)

logger = logging.getLogger(__name__)


class CrossAnalysisOutput(BaseModel):
    """Structured output from Sonnet cross-language analysis."""

    # Commonalities
    universal_themes: list[str] = Field(
        description="Themes appearing across most languages"
    )
    consensus_findings: list[str] = Field(
        description="Findings with broad agreement"
    )

    # Differences
    regional_variations: list[dict] = Field(
        description="List of {theme: str, variations: [{language: str, perspective: str}]}"
    )
    conflicting_findings: list[dict] = Field(
        description="List of {topic: str, conflicts: [{language: str, claim: str}]}"
    )
    unique_contributions: dict[str, list[str]] = Field(
        description="Map of language_code to list of unique insights"
    )

    # Coverage
    coverage_gaps_in_english: list[str] = Field(
        description="Topics better covered in non-English sources"
    )
    enhanced_areas: list[str] = Field(
        description="Areas where non-English sources added depth"
    )

    # Integration guidance
    integration_priority: list[str] = Field(
        description="Language codes ordered by value-add (most valuable first)"
    )
    synthesis_strategy: str = Field(
        description="Guidance for how Opus should integrate findings"
    )

    # The document itself
    comparative_document: str = Field(
        description="Full markdown comparative analysis document"
    )


def _format_language_findings(language_results: list[LanguageResult]) -> str:
    """Format all language results into structured string for prompt."""
    sections = []

    for result in language_results:
        section = f"### {result['language_name']} ({result['language_code']})\n"
        section += f"**Sources:** {result['source_count']}\n"
        section += f"**Workflows:** {', '.join(result['workflows_run'])}\n\n"

        section += "**Key Insights:**\n"
        for insight in result["key_insights"]:
            section += f"- {insight}\n"

        section += f"\n**Summary:**\n{result['findings_summary']}\n\n"
        section += "---\n"

        sections.append(section)

    return "\n".join(sections)


async def run_sonnet_analysis(state: MultiLangState) -> dict:
    """Run Sonnet 1M analysis across all language results."""
    try:
        language_results = state["language_results"]

        # Handle case where only English exists or no results
        if not language_results or (
            len(language_results) == 1 and language_results[0]["language_code"] == "en"
        ):
            logger.info("Only English sources available, skipping cross-language analysis")
            minimal_analysis: SonnetCrossAnalysis = {
                "universal_themes": [],
                "consensus_findings": [],
                "regional_variations": [],
                "conflicting_findings": [],
                "unique_contributions": {},
                "coverage_gaps_in_english": [],
                "enhanced_areas": [],
                "integration_priority": [],
                "synthesis_strategy": "Only English sources available - no cross-language integration needed.",
                "comparative_document": "# Comparative Analysis\n\nOnly English language sources were analyzed. No cross-language comparison available.",
            }
            return {
                "sonnet_analysis": minimal_analysis,
                "current_phase": "cross_language_analysis",
                "current_status": "Comparative analysis complete (English only)",
            }

        # Format findings
        topic = state["input"]["topic"]
        research_questions = state["input"].get("research_questions") or []
        questions_formatted = "\n".join(f"- {q}" for q in research_questions)

        language_findings = _format_language_findings(language_results)

        logger.debug(f"Running cross-language analysis for {len(language_results)} languages")

        user_prompt = CROSS_ANALYSIS_USER.format(
            topic=topic,
            research_questions=questions_formatted,
            language_findings=language_findings,
        )

        result: CrossAnalysisOutput = await get_structured_output(
            output_schema=CrossAnalysisOutput,
            user_prompt=user_prompt,
            system_prompt=CROSS_ANALYSIS_SYSTEM,
            tier=ModelTier.SONNET,
            max_tokens=16384,
        )

        # Remove English from integration priority if present
        integration_priority = [
            code for code in result.integration_priority if code != "en"
        ]

        # Convert to SonnetCrossAnalysis dict
        analysis: SonnetCrossAnalysis = {
            "universal_themes": result.universal_themes,
            "consensus_findings": result.consensus_findings,
            "regional_variations": result.regional_variations,
            "conflicting_findings": result.conflicting_findings,
            "unique_contributions": result.unique_contributions,
            "coverage_gaps_in_english": result.coverage_gaps_in_english,
            "enhanced_areas": result.enhanced_areas,
            "integration_priority": integration_priority,
            "synthesis_strategy": result.synthesis_strategy,
            "comparative_document": result.comparative_document,
        }

        logger.info("Cross-language analysis complete")

        return {
            "sonnet_analysis": analysis,
            "current_phase": "cross_language_analysis",
            "current_status": "Comparative analysis complete",
        }

    except Exception as e:
        logger.error(f"Cross-language analysis failed: {e}")
        error_dict = {
            "timestamp": datetime.now().isoformat(),
            "phase": "sonnet_analysis",
            "error": str(e),
            "error_type": type(e).__name__,
        }

        # Return partial analysis on error
        partial_analysis: SonnetCrossAnalysis = {
            "universal_themes": [],
            "consensus_findings": [],
            "regional_variations": [],
            "conflicting_findings": [],
            "unique_contributions": {},
            "coverage_gaps_in_english": [],
            "enhanced_areas": [],
            "integration_priority": [],
            "synthesis_strategy": "Analysis failed - see errors for details",
            "comparative_document": f"# Error\n\nCross-language analysis failed: {str(e)}",
        }

        return {
            "sonnet_analysis": partial_analysis,
            "current_phase": "cross_language_analysis",
            "current_status": f"Analysis failed: {str(e)}",
            "errors": [error_dict],
        }
