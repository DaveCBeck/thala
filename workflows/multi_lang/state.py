"""
State schemas for multi-language research workflow.

Defines TypedDict states for orchestrating research across multiple languages,
with relevance filtering, per-language quality settings, and cross-language synthesis.
"""

from datetime import datetime
from operator import add
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict


# =============================================================================
# Quality Configuration
# =============================================================================


class LanguageQualitySettings(TypedDict):
    """Quality settings for a specific language."""

    language_code: str
    quality_tier: Literal["quick", "standard", "comprehensive"]


class MultiLangQualitySettings(TypedDict):
    """Quality settings with per-language overrides."""

    default_quality: Literal["quick", "standard", "comprehensive"]
    per_language_overrides: dict[str, LanguageQualitySettings]


# =============================================================================
# Workflow Selection
# =============================================================================


class WorkflowSelection(TypedDict):
    """Which research workflows to run per language."""

    web: bool
    academic: bool
    books: bool


# =============================================================================
# Relevance and Results
# =============================================================================


class LanguageRelevanceCheck(TypedDict):
    """Haiku-powered relevance decision for a language."""

    language_code: str
    has_meaningful_discussion: bool
    confidence: float  # 0-1
    reasoning: str
    suggested_depth: Literal["skip", "quick", "standard", "comprehensive"]


class LanguageResult(TypedDict):
    """Results from research in one language."""

    language_code: str
    language_name: str
    started_at: datetime
    completed_at: Optional[datetime]
    workflows_run: list[str]  # ["web", "academic", "books"]
    quality_used: str
    findings_summary: str
    full_report: Optional[str]
    source_count: int
    key_insights: list[str]
    unique_perspectives: list[str]
    store_record_id: Optional[str]  # UUID as string
    errors: list[dict]


# =============================================================================
# Synthesis Types
# =============================================================================


class SonnetCrossAnalysis(TypedDict):
    """Sonnet-powered comparative analysis across languages."""

    # Commonalities
    universal_themes: list[str]
    consensus_findings: list[str]
    # Differences
    regional_variations: list[dict]  # {theme, variations: [{language, perspective}]}
    conflicting_findings: list[dict]
    unique_contributions: dict[str, list[str]]  # language_code -> insights
    # Coverage
    coverage_gaps_in_english: list[str]
    enhanced_areas: list[str]
    # For Opus
    integration_priority: list[str]  # ordered language codes
    synthesis_strategy: str
    # Formatted output
    comparative_document: str  # The actual markdown document


class OpusIntegrationStep(TypedDict):
    """Opus-powered integration of one language into synthesis."""

    language_code: str
    language_name: str
    integrated_content: str
    enhancement_notes: str
    new_sections_added: list[str]
    existing_sections_enhanced: list[str]


# =============================================================================
# Input Types
# =============================================================================


class MultiLangInput(TypedDict):
    """Input parameters for multi-language research workflow."""

    topic: str
    research_questions: Optional[list[str]]
    brief: Optional[str]
    mode: Literal["set_languages", "all_languages"]
    languages: Optional[list[str]]  # ISO 639-1 codes for set_languages mode
    extend_to_all_30: bool
    workflows: WorkflowSelection
    quality_settings: MultiLangQualitySettings


# =============================================================================
# Checkpoint Types
# =============================================================================


class CheckpointPhase(TypedDict):
    """Tracks which phases have completed for resumption."""

    language_selection: bool
    relevance_checks: bool
    languages_executed: dict[str, bool]  # code -> completed
    sonnet_analysis: bool
    opus_integration: bool
    saved_to_store: bool


# =============================================================================
# Custom Reducers
# =============================================================================


def merge_language_results(
    existing: list[LanguageResult], new: list[LanguageResult]
) -> list[LanguageResult]:
    """Merge language results, keeping latest by language_code."""
    merged = {r["language_code"]: r for r in existing}
    for result in new:
        merged[result["language_code"]] = result
    return list(merged.values())


# =============================================================================
# Main State
# =============================================================================


class MultiLangState(TypedDict):
    """Complete state for multi-language research orchestration."""

    # Input
    input: MultiLangInput

    # Language tracking
    target_languages: list[str]
    language_configs: dict  # code -> LanguageConfig dict
    languages_with_content: list[str]  # After relevance filtering
    current_language_index: int
    languages_completed: list[str]
    languages_failed: list[str]

    # Relevance checks (mode 2)
    relevance_checks: Annotated[list[LanguageRelevanceCheck], add]

    # Per-language results
    language_results: Annotated[list[LanguageResult], merge_language_results]

    # Synthesis outputs
    sonnet_analysis: Optional[SonnetCrossAnalysis]
    integration_steps: Annotated[list[OpusIntegrationStep], add]
    final_synthesis: Optional[str]  # Synthesized document

    # Store IDs
    per_language_record_ids: dict[str, str]
    comparative_record_id: Optional[str]
    synthesis_record_id: Optional[str]

    # Checkpointing
    checkpoint_phase: CheckpointPhase
    checkpoint_path: Optional[str]

    # Metadata
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    current_phase: str
    current_status: str
    langsmith_run_id: Optional[str]
    errors: Annotated[list[dict], add]
