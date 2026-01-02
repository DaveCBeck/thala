"""Language executor node for multi-lingual research workflow."""

import logging
from datetime import datetime
from typing import Optional

from workflows.multi_lang.state import MultiLangState, LanguageResult
from workflows.shared.llm_utils.models import get_llm, ModelTier

logger = logging.getLogger(__name__)


async def _run_web_research(
    topic: str,
    language_config: dict,
    quality: str,
    research_questions: list[str] | None = None,
) -> dict:
    """Run deep_research workflow in the target language."""
    from workflows.research.graph.api import deep_research

    try:
        result = await deep_research(
            query=topic,
            depth=quality,
            language=language_config["code"],
        )

        return {
            "final_report": result.get("final_report"),
            "source_count": len(result.get("research_findings", [])),
            "status": "completed" if result.get("final_report") else "failed",
            "errors": result.get("errors", []),
        }
    except Exception as e:
        logger.error(f"Web research failed for {language_config['code']}: {e}")
        return {
            "final_report": None,
            "source_count": 0,
            "status": "failed",
            "errors": [{"phase": "web_research", "error": str(e)}],
        }


async def _run_academic_research(
    topic: str,
    language_config: dict,
    quality: str,
    research_questions: list[str] | None = None,
) -> dict:
    """Run academic_lit_review in the target language."""
    from workflows.research.subgraphs.academic_lit_review.graph.api import (
        academic_lit_review,
    )

    try:
        # Default questions if not provided
        questions = research_questions or [
            f"What is the current state of research on {topic}?",
            f"What are the key findings and debates about {topic}?",
        ]

        result = await academic_lit_review(
            topic=topic,
            research_questions=questions,
            quality=quality,
            language=language_config["code"],
        )

        return {
            "final_report": result.get("final_review"),
            "source_count": len(result.get("paper_corpus", {})),
            "status": "completed" if result.get("final_review") else "failed",
            "errors": result.get("errors", []),
        }
    except Exception as e:
        logger.error(f"Academic research failed for {language_config['code']}: {e}")
        return {
            "final_report": None,
            "source_count": 0,
            "status": "failed",
            "errors": [{"phase": "academic_research", "error": str(e)}],
        }


async def _run_book_research(
    topic: str,
    language_config: dict,
    quality: str,
) -> dict:
    """Run book_finding in the target language."""
    from workflows.research.subgraphs.book_finding.graph.api import book_finding

    try:
        result = await book_finding(
            theme=topic,
            quality=quality,
            language=language_config["code"],
        )

        return {
            "final_report": result.get("final_markdown"),
            "source_count": len(result.get("processed_books", [])),
            "status": "completed" if result.get("final_markdown") else "failed",
            "errors": result.get("errors", []),
        }
    except Exception as e:
        logger.error(f"Book research failed for {language_config['code']}: {e}")
        return {
            "final_report": None,
            "source_count": 0,
            "status": "failed",
            "errors": [{"phase": "book_research", "error": str(e)}],
        }


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

    prompt = f"""You are analyzing research findings in {language_config['name']}.

Consolidate the following research outputs into a comprehensive summary:

{combined_reports}

Provide:
1. A concise summary (2-3 paragraphs) of the main findings
2. 3-5 key insights that are particularly important
3. 2-4 perspectives or viewpoints that appear unique to {language_config['name']} sources

Format your response as JSON:
{{
    "summary": "...",
    "key_insights": ["...", "..."],
    "unique_perspectives": ["...", "..."]
}}"""

    try:
        response = await llm.ainvoke(prompt)
        content = response.content

        # Parse JSON response
        import json

        data = json.loads(content)
        return (
            data.get("summary", ""),
            data.get("key_insights", []),
            data.get("unique_perspectives", []),
        )
    except Exception as e:
        logger.error(f"Failed to compress findings: {e}")
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
    languages_to_process = state.get("languages_with_content") or state["target_languages"]

    if idx >= len(languages_to_process):
        # Should not happen - router should prevent this
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

    logger.info(f"Executing workflows for {language_name} ({language_code})")

    started_at = datetime.utcnow()

    # Get quality settings
    quality_settings = state["input"]["quality_settings"]
    per_lang_overrides = quality_settings.get("per_language_overrides", {})

    if language_code in per_lang_overrides:
        quality = per_lang_overrides[language_code]["quality_tier"]
    else:
        quality = quality_settings["default_quality"]

    # Determine which workflows to run
    workflows = state["input"]["workflows"]
    topic = state["input"]["topic"]
    research_questions = state["input"].get("research_questions")

    workflow_results = []
    total_sources = 0
    all_errors = []

    # Run web research
    if workflows.get("web", False):
        logger.info(f"Running web research for {language_name}")
        result = await _run_web_research(topic, language_config, quality, research_questions)
        workflow_results.append({
            "workflow_type": "Web Research",
            "report": result["final_report"],
            "source_count": result["source_count"],
            "status": result["status"],
        })
        total_sources += result["source_count"]
        all_errors.extend(result.get("errors", []))

    # Run academic research
    if workflows.get("academic", False):
        logger.info(f"Running academic research for {language_name}")
        result = await _run_academic_research(topic, language_config, quality, research_questions)
        workflow_results.append({
            "workflow_type": "Academic Research",
            "report": result["final_report"],
            "source_count": result["source_count"],
            "status": result["status"],
        })
        total_sources += result["source_count"]
        all_errors.extend(result.get("errors", []))

    # Run book research
    if workflows.get("books", False):
        logger.info(f"Running book research for {language_name}")
        result = await _run_book_research(topic, language_config, quality)
        workflow_results.append({
            "workflow_type": "Book Research",
            "report": result["final_report"],
            "source_count": result["source_count"],
            "status": result["status"],
        })
        total_sources += result["source_count"]
        all_errors.extend(result.get("errors", []))

    # Check if any workflow succeeded
    has_results = any(r["status"] == "completed" for r in workflow_results)

    if not has_results:
        # All workflows failed
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
            ] + all_errors,
        }

    # Compress findings
    summary, key_insights, unique_perspectives = await _compress_language_findings(
        workflow_results, language_config
    )

    # Get full report (combine all successful reports)
    full_report = "\n\n".join(
        r["report"] for r in workflow_results if r["report"]
    )

    # Track which workflows ran
    workflows_run = []
    if workflows.get("web"):
        workflows_run.append("web")
    if workflows.get("academic"):
        workflows_run.append("academic")
    if workflows.get("books"):
        workflows_run.append("books")

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
        store_record_id=None,  # Will be set by store_results node
        errors=all_errors,
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
    languages_to_process = state.get("languages_with_content") or state["target_languages"]
    total = len(languages_to_process)

    if idx >= total:
        return {
            "current_phase": "languages_complete",
            "current_status": f"All languages complete ({total}/{total})",
        }
    else:
        return {
            "current_status": f"Processed {idx}/{total} languages",
        }
