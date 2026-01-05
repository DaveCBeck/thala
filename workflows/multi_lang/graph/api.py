"""
API entry point for multi_lang research workflow.

Provides the main function for running multi-language research.
"""

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from workflows.multi_lang.state import (
    MultiLangState,
    MultiLangInput,
    MultiLangQualitySettings,
    CheckpointPhase,
)
from workflows.multi_lang.graph.construction import multi_lang_graph
from workflows.multi_lang.workflow_registry import build_workflow_selection


class MultiLangResult:
    """Result from multi_lang_research workflow."""

    def __init__(self, state: MultiLangState):
        self.synthesis = state.get("final_synthesis")
        self.comparative = state.get("sonnet_analysis", {}).get("comparative_document") if state.get("sonnet_analysis") else None
        self.language_results = state.get("language_results", [])
        self.sonnet_analysis = state.get("sonnet_analysis")
        self.integration_steps = state.get("integration_steps", [])
        self.synthesis_record_id = state.get("synthesis_record_id")
        self.comparative_record_id = state.get("comparative_record_id")
        self.per_language_record_ids = state.get("per_language_record_ids", {})
        self.started_at = state.get("started_at")
        self.completed_at = state.get("completed_at")
        self.errors = state.get("errors", [])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "synthesis": self.synthesis,
            "comparative": self.comparative,
            "language_results": self.language_results,
            "sonnet_analysis": self.sonnet_analysis,
            "integration_steps": self.integration_steps,
            "synthesis_record_id": self.synthesis_record_id,
            "comparative_record_id": self.comparative_record_id,
            "per_language_record_ids": self.per_language_record_ids,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "errors": self.errors,
        }


async def multi_lang_research(
    topic: str,
    mode: Literal["set_languages", "all_languages"] = "set_languages",
    languages: Optional[list[str]] = None,
    research_questions: Optional[list[str]] = None,
    brief: Optional[str] = None,
    workflows: Optional[dict[str, bool]] = None,
    quality: Literal["quick", "standard", "comprehensive"] = "standard",
    per_language_quality: Optional[dict[str, dict]] = None,
    extend_to_all_30: bool = False,
) -> MultiLangResult:
    """
    Run multi-language research workflow.

    Orchestrates research across multiple languages with relevance filtering,
    per-language quality settings, and cross-language synthesis.

    Args:
        topic: The research topic
        mode: "set_languages" (specify exact languages) or "all_languages" (auto-detect)
        languages: ISO 639-1 codes for set_languages mode (e.g., ["en", "es", "de"])
        research_questions: Optional specific questions to investigate
        brief: Optional additional context for the research
        workflows: Which workflows to run {"web": True, "academic": False, "books": False}
        quality: Default quality tier for all languages
        per_language_quality: Override quality per language {"es": {"quality_tier": "comprehensive"}}
        extend_to_all_30: For all_languages mode, extend beyond major 10?

    Returns:
        MultiLangResult with:
        - synthesis: Unified integrated English report
        - comparative: Cross-language analysis document
        - language_results: Per-language findings
        - sonnet_analysis: Structured analysis data
        - Various store record IDs

    Examples:
        # Research in specific languages
        result = await multi_lang_research(
            topic="sustainable urban planning",
            mode="set_languages",
            languages=["en", "es", "de", "ja"],
            workflows={"web": True, "academic": True, "books": False},
        )

        # Auto-detect relevant languages
        result = await multi_lang_research(
            topic="traditional medicine practices",
            mode="all_languages",
            extend_to_all_30=True,
            workflows={"web": True, "academic": False, "books": True},
        )
    """
    # Validate mode and languages
    if mode == "set_languages" and not languages:
        languages = ["en"]

    # Build workflow selection from registry (applies user overrides to defaults)
    workflow_selection = build_workflow_selection(workflows)

    # Build quality settings
    quality_settings = MultiLangQualitySettings(
        default_quality=quality,
        per_language_overrides=per_language_quality or {},
    )

    # Build input
    input_data = MultiLangInput(
        topic=topic,
        research_questions=research_questions,
        brief=brief,
        mode=mode,
        languages=languages,
        extend_to_all_30=extend_to_all_30,
        workflows=workflow_selection,
        quality_settings=quality_settings,
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
        "checkpoint_phase": CheckpointPhase(
            language_selection=False,
            relevance_checks=False,
            languages_executed={},
            sonnet_analysis=False,
            opus_integration=False,
            saved_to_store=False,
        ),
        "checkpoint_path": None,
        "started_at": datetime.utcnow(),
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
    }

    result_state = await multi_lang_graph.ainvoke(initial_state, config=config)

    return MultiLangResult(result_state)
