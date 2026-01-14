"""Language executor node for multi-lingual research workflow.

Uses the workflow registry for pluggable workflow dispatch.
"""

import logging
from datetime import datetime

from workflows.wrappers.multi_lang.state import MultiLangState, LanguageResult
from workflows.wrappers.multi_lang.workflow_registry import WORKFLOW_REGISTRY
from workflows.shared.llm_utils.models import get_llm, ModelTier
from workflows.research.web_research.utils import extract_json_from_llm_response

logger = logging.getLogger(__name__)


async def _compress_language_findings(
    workflow_results: list[dict],
    language_config: dict,
) -> tuple[str, list[str], list[str]]:
    """
    Compress all workflow results for a language into summary.

    Uses Sonnet to:
    1. Create a concise summary of all findings
    2. Extract 3-5 key insights
    3. Identify perspectives unique to this language

    Returns: (summary, key_insights, unique_perspectives)
    """
    llm = get_llm(ModelTier.SONNET, max_tokens=4096)

    # Combine all reports
    combined_reports = "\n\n---\n\n".join(
        f"## {result['workflow_type']}\n\n{result['report']}"
        for result in workflow_results
        if result.get("report")
    )

    if not combined_reports:
        return ("No findings to compress", [], [])

    prompt = f"""You are analyzing research findings in {language_config["name"]}.

Consolidate the following research outputs into a comprehensive summary:

{combined_reports}

Provide:
1. A concise summary (2-3 paragraphs) of the main findings
2. 3-5 key insights that are particularly important
3. 2-4 perspectives or viewpoints that appear unique to {language_config["name"]} sources

Format your response as JSON:
{{
    "summary": "...",
    "key_insights": ["...", "..."],
    "unique_perspectives": ["...", "..."]
}}"""

    try:
        response = await llm.ainvoke(prompt)
        content = response.content

        # Parse JSON response using robust extraction
        data = extract_json_from_llm_response(content)
        return (
            data.get("summary", ""),
            data.get("key_insights", []),
            data.get("unique_perspectives", []),
        )
    except Exception as e:
        logger.error(f"Failed to compress findings for {language_config['name']}: {e}")
        return (
            f"Summary generation failed: {e}",
            ["Error compressing insights"],
            [],
        )


async def execute_next_language(state: MultiLangState) -> dict:
    """
    Execute chosen workflow(s) for the current language in the sequence.

    Uses current_language_index to determine which language to process.
    Checks languages_with_content (if set) or target_languages.

    For the current language:
    1. Get quality settings (check per_language_overrides)
    2. Run selected workflows (web, academic, books based on input.workflows)
    3. Compress findings into summary with key insights
    4. Create LanguageResult

    Returns:
        language_results: [LanguageResult]  # appends via reducer
        current_language_index: incremented
        current_phase: f"executing_{language_code}"
        current_status: f"Completed {language_name} ({N} sources)"

    On error:
        languages_failed: [language_code]
        errors: [{language, error, phase}]
    """
    idx = state["current_language_index"]

    # Determine which language list to use
    languages_to_process = (
        state.get("languages_with_content") or state["target_languages"]
    )

    if idx >= len(languages_to_process):
        logger.error(
            f"Language index {idx} exceeds list length {len(languages_to_process)}"
        )
        return {
            "current_status": "Error: Language index out of bounds",
            "errors": [
                {
                    "phase": "language_execution",
                    "error": f"Index {idx} exceeds language list length {len(languages_to_process)}",
                }
            ],
        }

    language_code = languages_to_process[idx]
    language_config = state["language_configs"][language_code]
    language_name = language_config["name"]

    logger.info(f"Executing workflow for {language_name}")

    started_at = datetime.utcnow()

    # Get quality (single global quality tier)
    quality = state["input"]["quality"]
    logger.debug(f"Using quality={quality} for {language_name}")

    # Get the single workflow to run
    workflow_key = state["input"]["workflow"]
    topic = state["input"]["topic"]
    research_questions = state["input"].get("research_questions")

    workflow_results = []
    workflows_run = []
    total_sources = 0
    all_errors = []

    # Get workflow config from registry
    config = WORKFLOW_REGISTRY.get(workflow_key)
    if not config:
        logger.error(f"Unknown workflow: {workflow_key}")
        return {
            "languages_failed": [language_code],
            "current_language_index": idx + 1,
            "current_phase": f"failed_{language_code}",
            "current_status": f"Failed: Unknown workflow '{workflow_key}'",
            "errors": [
                {
                    "language": language_code,
                    "phase": "language_execution",
                    "error": f"Unknown workflow: {workflow_key}",
                }
            ],
        }

    logger.debug(f"Running {config['name']} for {language_name}")

    # Build arguments for the workflow runner
    kwargs = {
        "topic": topic,
        "language_config": language_config,
        "quality": quality,
    }
    if config["requires_questions"]:
        kwargs["research_questions"] = research_questions

    # Run the workflow
    result = await config["runner"](**kwargs)

    workflow_results.append(
        {
            "workflow_type": config["name"],
            "report": result["final_report"],
            "source_count": result["source_count"],
            "status": result["status"],
        }
    )
    workflows_run.append(workflow_key)
    total_sources += result["source_count"]
    all_errors.extend(result.get("errors", []))

    # Check if any workflow succeeded
    has_results = any(
        r["status"] in ("success", "partial", "completed") for r in workflow_results
    )

    if not has_results:
        logger.error(f"All workflows failed for {language_name}")
        return {
            "languages_failed": [language_code],
            "current_language_index": idx + 1,
            "current_phase": f"failed_{language_code}",
            "current_status": f"Failed: {language_name} (all workflows failed)",
            "errors": [
                {
                    "language": language_code,
                    "phase": "language_execution",
                    "error": f"All workflows failed for {language_name}",
                }
            ]
            + all_errors,
        }

    # Check for zero results
    if total_sources == 0:
        logger.info(
            f"No sources found for {language_name} - workflows ran but returned no results"
        )
        return {
            "languages_completed": [language_code],
            "current_language_index": idx + 1,
            "current_phase": f"completed_{language_code}",
            "current_status": f"Completed {language_name} (no sources found)",
        }

    # Compress findings
    logger.debug(f"Compressing findings for {language_name}")
    summary, key_insights, unique_perspectives = await _compress_language_findings(
        workflow_results, language_config
    )

    # Get full report (combine all successful reports)
    full_report = "\n\n".join(r["report"] for r in workflow_results if r["report"])

    # Create LanguageResult
    language_result = LanguageResult(
        language_code=language_code,
        language_name=language_name,
        started_at=started_at,
        completed_at=datetime.utcnow(),
        workflows_run=workflows_run,
        quality_used=quality,
        findings_summary=summary,
        full_report=full_report,
        source_count=total_sources,
        key_insights=key_insights,
        unique_perspectives=unique_perspectives,
        store_record_id=None,
        errors=all_errors,
    )

    logger.info(
        f"Completed {language_name}: {total_sources} sources from {len(workflows_run)} workflows"
    )

    return {
        "language_results": [language_result],
        "languages_completed": [language_code],
        "current_language_index": idx + 1,
        "current_phase": f"executed_{language_code}",
        "current_status": f"Completed {language_name} ({total_sources} sources)",
    }


async def check_languages_complete(state: MultiLangState) -> dict:
    """
    Check if all languages have been processed.

    This is a pass-through node that the router uses to decide
    whether to loop back or proceed to synthesis.

    Returns:
        current_status: "Processed X/Y languages" or "All languages complete"
    """
    idx = state["current_language_index"]
    languages_to_process = (
        state.get("languages_with_content") or state["target_languages"]
    )
    total = len(languages_to_process)

    if idx >= total:
        logger.info(f"All {total} languages complete")
        return {
            "current_phase": "languages_complete",
            "current_status": f"All languages complete ({total}/{total})",
        }
    else:
        logger.debug(f"Processed {idx}/{total} languages")
        return {
            "current_status": f"Processed {idx}/{total} languages",
        }
