"""Opus-powered one-by-one integration producing synthesized documents."""

import logging
from datetime import datetime
from pydantic import BaseModel, Field

from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.wrappers.multi_lang.prompts.integration import (
    INITIAL_SYNTHESIS_SYSTEM,
    INITIAL_SYNTHESIS_USER,
    INTEGRATION_SYSTEM,
    INTEGRATION_USER,
    FINAL_ENHANCEMENT_SYSTEM,
    FINAL_ENHANCEMENT_USER,
)
from workflows.wrappers.multi_lang.state import (
    MultiLangState,
    LanguageResult,
    OpusIntegrationStep,
)

logger = logging.getLogger(__name__)


class InitialSynthesisOutput(BaseModel):
    """Structured output from initial synthesis creation."""

    synthesis_document: str = Field(
        description="The initial synthesis document from English findings"
    )


class IntegrationOutput(BaseModel):
    """Structured output from integrating one language."""

    updated_document: str = Field(
        description="The synthesis document with new language integrated"
    )
    enhancement_notes: str = Field(
        description="What was added or changed during integration"
    )
    new_sections_added: list[str] = Field(
        description="List of new section titles added"
    )
    existing_sections_enhanced: list[str] = Field(
        description="List of existing section titles that were enhanced"
    )


class FinalEnhancementOutput(BaseModel):
    """Structured output from final document enhancement."""

    finalized_document: str = Field(
        description="The polished, finalized synthesis document"
    )


async def _create_initial_synthesis(
    english_result: LanguageResult,
    topic: str,
    research_questions: list[str] | None,
) -> str:
    """Create the initial synthesis document from English findings."""
    logger.debug("Creating initial synthesis from English findings")

    questions_formatted = (
        "\n".join(f"- {q}" for q in research_questions)
        if research_questions
        else "None specified"
    )

    user_prompt = INITIAL_SYNTHESIS_USER.format(
        topic=topic,
        research_questions=questions_formatted,
        english_findings=english_result["findings_summary"],
    )

    result: InitialSynthesisOutput = await get_structured_output(
        output_schema=InitialSynthesisOutput,
        user_prompt=user_prompt,
        system_prompt=INITIAL_SYNTHESIS_SYSTEM,
        tier=ModelTier.OPUS,
        max_tokens=16384,
    )

    return result.synthesis_document


async def _integrate_language(
    current_document: str,
    language_result: LanguageResult,
    sonnet_guidance: list[str],
) -> tuple[str, OpusIntegrationStep]:
    """Integrate one language's findings into the current synthesis."""
    language_name = language_result["language_name"]
    language_code = language_result["language_code"]

    logger.debug(f"Integrating {language_name} findings")

    guidance_formatted = "\n".join(f"- {item}" for item in sonnet_guidance)

    system_prompt = INTEGRATION_SYSTEM.format(language_name=language_name)
    user_prompt = INTEGRATION_USER.format(
        current_document=current_document,
        language_name=language_name,
        sonnet_guidance=guidance_formatted,
        language_findings=language_result["findings_summary"],
    )

    result: IntegrationOutput = await get_structured_output(
        output_schema=IntegrationOutput,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        tier=ModelTier.OPUS,
        max_tokens=16384,
        thinking_budget=8000,
    )

    integration_step: OpusIntegrationStep = {
        "language_code": language_code,
        "language_name": language_name,
        "integrated_content": result.updated_document,
        "enhancement_notes": result.enhancement_notes,
        "new_sections_added": result.new_sections_added,
        "existing_sections_enhanced": result.existing_sections_enhanced,
    }

    return result.updated_document, integration_step


async def _finalize_synthesis(
    current_document: str,
    languages_integrated: list[str],
    workflows_used: list[str],
    integration_steps: list[OpusIntegrationStep],
) -> str:
    """Finalize the synthesis document."""
    logger.debug("Finalizing synthesis document")

    languages_list = ", ".join(languages_integrated)
    workflows_list = ", ".join(workflows_used)

    integration_notes = []
    for step in integration_steps:
        note = f"**{step['language_name']}:**\n{step['enhancement_notes']}"
        integration_notes.append(note)
    integration_notes_formatted = "\n\n".join(integration_notes)

    user_prompt = FINAL_ENHANCEMENT_USER.format(
        current_document=current_document,
        languages_list=languages_list,
        workflows_list=workflows_list,
        integration_notes=integration_notes_formatted,
    )

    result: FinalEnhancementOutput = await get_structured_output(
        output_schema=FinalEnhancementOutput,
        user_prompt=user_prompt,
        system_prompt=FINAL_ENHANCEMENT_SYSTEM,
        tier=ModelTier.OPUS,
        max_tokens=16384,
    )

    return result.finalized_document


async def run_opus_integration(state: MultiLangState) -> dict:
    """Opus integrates findings one language at a time."""
    try:
        language_results = state["language_results"]
        sonnet_analysis = state.get("sonnet_analysis")

        if not language_results:
            logger.warning("No language results available for integration")
            return {
                "integration_steps": [],
                "final_synthesis": "No language results available for integration.",
                "current_phase": "opus_integration",
                "current_status": "Integration skipped - no results",
            }

        # Find English result or use first available
        english_result = next(
            (r for r in language_results if r["language_code"] == "en"), None
        )
        baseline_result = english_result or language_results[0]

        # Create initial synthesis
        topic = state["input"]["topic"]
        research_questions = state["input"].get("research_questions")

        logger.info("Starting Opus integration")

        current_document = await _create_initial_synthesis(
            baseline_result, topic, research_questions
        )

        # Get integration priority from Sonnet (excluding English)
        integration_priority = []
        if sonnet_analysis and sonnet_analysis.get("integration_priority"):
            integration_priority = [
                code
                for code in sonnet_analysis["integration_priority"]
                if code != "en"
            ]

        # Build language code -> result mapping
        results_by_code = {r["language_code"]: r for r in language_results}

        # Track integration steps
        integration_steps: list[OpusIntegrationStep] = []

        # Integrate each language in priority order
        for language_code in integration_priority:
            if language_code not in results_by_code:
                continue

            language_result = results_by_code[language_code]

            # Get unique contributions from Sonnet
            sonnet_guidance = []
            if sonnet_analysis and sonnet_analysis.get("unique_contributions"):
                sonnet_guidance = sonnet_analysis["unique_contributions"].get(
                    language_code, []
                )

            try:
                current_document, step = await _integrate_language(
                    current_document, language_result, sonnet_guidance
                )
                integration_steps.append(step)
                logger.debug(f"Integrated {language_result['language_name']}")

            except Exception as e:
                logger.error(f"Failed to integrate {language_code}: {e}")
                error_dict = {
                    "timestamp": datetime.now().isoformat(),
                    "phase": "opus_integration",
                    "language": language_code,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                continue

        # Finalize the document
        languages_integrated = [baseline_result["language_name"]] + [
            step["language_name"] for step in integration_steps
        ]

        # Collect all unique workflows used
        workflows_used = list(
            set(
                workflow
                for result in language_results
                for workflow in result["workflows_run"]
            )
        )

        final_document = await _finalize_synthesis(
            current_document, languages_integrated, workflows_used, integration_steps
        )

        logger.info(f"Integration complete: {len(languages_integrated)} languages integrated")

        return {
            "integration_steps": integration_steps,
            "final_synthesis": final_document,
            "current_phase": "opus_integration",
            "current_status": "Synthesis complete",
        }

    except Exception as e:
        logger.error(f"Opus integration failed: {e}")
        error_dict = {
            "timestamp": datetime.now().isoformat(),
            "phase": "opus_integration",
            "error": str(e),
            "error_type": type(e).__name__,
        }

        return {
            "integration_steps": [],
            "final_synthesis": f"Integration failed: {str(e)}",
            "current_phase": "opus_integration",
            "current_status": f"Integration failed: {str(e)}",
            "errors": [error_dict],
        }
