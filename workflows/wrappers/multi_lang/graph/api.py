"""
API entry point for multi_lang research workflow.

Provides the main function for running multi-language research.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langsmith import traceable

from workflows.shared.quality_config import QualityTier
from workflows.wrappers.multi_lang.state import (
    MultiLangState,
    MultiLangInput,
    LanguageMode,
    WorkflowType,
)
from workflows.wrappers.multi_lang.graph.construction import multi_lang_graph
from workflows.shared.workflow_state_store import save_workflow_state


class MultiLangResult:
    """Result from multi_lang_research workflow.

    Contains only standardized fields. For detailed state (language_results,
    sonnet_analysis, etc.), use load_workflow_state("multi_lang", langsmith_run_id).
    """

    def __init__(self, state: MultiLangState):
        synthesis = state.get("final_synthesis")
        errors = state.get("errors", [])

        # Standardized fields only
        self.final_report = synthesis
        self.langsmith_run_id = state.get("langsmith_run_id")
        self.errors = errors
        self.source_count = len(state.get("language_results", []))
        self.started_at = state.get("started_at")
        self.completed_at = state.get("completed_at")

        # Determine status
        if synthesis and not errors:
            self.status = "success"
        elif synthesis and errors:
            self.status = "partial"
        else:
            self.status = "failed"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_report": self.final_report,
            "status": self.status,
            "langsmith_run_id": self.langsmith_run_id,
            "errors": self.errors,
            "source_count": self.source_count,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@traceable(run_type="chain", name="MultiLangResearch")
async def multi_lang_research(
    topic: str,
    mode: LanguageMode = "set_languages",
    languages: Optional[list[str]] = None,
    research_questions: Optional[list[str]] = None,
    brief: Optional[str] = None,
    workflow: WorkflowType = "web",
    quality: QualityTier = "standard",
) -> MultiLangResult:
    """
    Run multi-language research workflow.

    Orchestrates research across multiple languages with relevance filtering
    and cross-language synthesis.

    Args:
        topic: The research topic
        mode: Language selection mode:
            - "set_languages": Use user-specified languages (no relevance filtering)
            - "main_languages": Use major 10 languages with relevance filtering
            - "all_languages": Use all 29 supported languages with relevance filtering
        languages: ISO 639-1 codes for set_languages mode (e.g., ["en", "es", "de"])
        research_questions: Optional specific questions to investigate
        brief: Optional additional context for the research
        workflow: Which workflow to run ("web", "academic", or "books")
        quality: Quality tier for all languages (test, quick, standard, comprehensive, high_quality)

    Returns:
        MultiLangResult with:
        - final_report: Unified integrated English report
        - status: "success", "partial", or "failed"
        - langsmith_run_id: For loading detailed state

    Examples:
        # Research in specific languages
        result = await multi_lang_research(
            topic="sustainable urban planning",
            mode="set_languages",
            languages=["en", "es", "de", "ja"],
            workflow="web",
        )

        # Auto-detect from major 10 languages
        result = await multi_lang_research(
            topic="traditional medicine practices",
            mode="main_languages",
            workflow="academic",
        )

        # Use all supported languages
        result = await multi_lang_research(
            topic="climate policy",
            mode="all_languages",
            workflow="web",
        )
    """
    # Validate mode and languages
    if mode == "set_languages" and not languages:
        languages = ["en"]

    # Build input
    input_data = MultiLangInput(
        topic=topic,
        research_questions=research_questions,
        brief=brief,
        mode=mode,
        languages=languages,
        workflow=workflow,
        quality=quality,
    )

    # Build initial state
    run_id = str(uuid.uuid4())
    initial_state: MultiLangState = {
        "input": input_data,
        "target_languages": [],
        "language_configs": {},
        "languages_with_content": [],
        "current_language_index": 0,
        "languages_completed": [],
        "languages_failed": [],
        "relevance_checks": [],
        "language_results": [],
        "sonnet_analysis": None,
        "integration_steps": [],
        "final_synthesis": None,
        "per_language_record_ids": {},
        "comparative_record_id": None,
        "synthesis_record_id": None,
        "started_at": datetime.now(timezone.utc),
        "completed_at": None,
        "current_phase": "initializing",
        "current_status": "Starting multi-language research",
        "langsmith_run_id": run_id,
        "errors": [],
    }

    # Run the graph
    config = {
        "run_id": run_id,
        "run_name": f"multi_lang:{topic[:50]}",
        "tags": [
            f"quality:{quality}",
            "workflow:multi_lang",
            f"mode:{mode}",
            f"subworkflow:{workflow}",
        ],
        "metadata": {
            "topic": topic[:100],
            "quality_tier": quality,
            "mode": mode,
            "workflow_type": workflow,
            "language_count": len(languages) if languages else 0,
        },
    }

    result_state = await multi_lang_graph.ainvoke(initial_state, config=config)

    # Save full state for downstream workflows (in dev/test mode)
    save_workflow_state(
        workflow_name="multi_lang",
        run_id=run_id,
        state={
            "input": dict(input_data) if hasattr(input_data, "_asdict") else input_data,
            "language_results": result_state.get("language_results", []),
            "sonnet_analysis": result_state.get("sonnet_analysis"),
            "integration_steps": result_state.get("integration_steps", []),
            "final_synthesis": result_state.get("final_synthesis"),
            "per_language_record_ids": result_state.get("per_language_record_ids", {}),
            "synthesis_record_id": result_state.get("synthesis_record_id"),
            "comparative_record_id": result_state.get("comparative_record_id"),
            "started_at": result_state.get("started_at"),
            "completed_at": result_state.get("completed_at"),
        },
    )

    return MultiLangResult(result_state)
