"""Transparency data collection for honest methodology reporting.

Aggregates real pipeline metrics from AcademicLitReviewState into a
structured TransparencyReport. Pure data extraction — no LLM calls.

All field accesses use .get() with defaults for compatibility with
older workflow states loaded from incremental checkpoints.
"""

import logging
from collections import Counter

from workflows.research.academic_lit_review.state import AcademicLitReviewState
from .types import DiffusionStageReport, TransparencyReport

# Hardcoded relevance threshold — matches keyword_search/searcher.py:197
# TODO: extract from QualitySettings when relevance_threshold is added there
_RELEVANCE_THRESHOLD = 0.6

_CLUSTERING_RATIONALE_MAX_LEN = 300


logger = logging.getLogger(__name__)


def _sanitise_for_template(value: str) -> str:
    """Escape curly braces and XML closing tags in LLM-derived content."""
    value = value.replace("{", "{{").replace("}", "}}")
    value = value.replace("</transparency_data>", "&lt;/transparency_data&gt;")
    return value


def collect_transparency_report(state: AcademicLitReviewState) -> TransparencyReport:
    """Aggregate transparency data from workflow state into a structured report.

    Args:
        state: The full AcademicLitReviewState after all phases have run.

    Returns:
        TransparencyReport with real pipeline metrics for methodology writing.
    """
    quality_settings = state.get("quality_settings", {})
    diffusion = state.get("diffusion", {})
    paper_summaries = state.get("paper_summaries", {})

    # Discovery counts
    keyword_paper_count = len(state.get("keyword_papers", []))
    citation_paper_count = len(state.get("citation_papers", []))
    expert_paper_count = len(state.get("expert_papers", []))

    # Diffusion stage reports
    # DiffusionStage.new_relevant/new_rejected are DOI lists — we take len()
    diffusion_stages: list[DiffusionStageReport] = []
    for stage in diffusion.get("stages", []):
        diffusion_stages.append(
            DiffusionStageReport(
                stage_number=stage.get("stage_number", 0),
                forward_found=stage.get("forward_papers_found", 0),
                backward_found=stage.get("backward_papers_found", 0),
                new_relevant=len(stage.get("new_relevant", [])),
                new_rejected=len(stage.get("new_rejected", [])),
                coverage_delta=stage.get("coverage_delta", 0.0),
            )
        )

    # Fallback substitutions summary
    fallback_subs = state.get("fallback_substitutions", [])
    fallback_summary = _format_fallback_summary(fallback_subs)

    # Corpus date range from paper summaries
    years = [s.get("year", 0) for s in paper_summaries.values() if s.get("year")]
    date_range = f"{min(years)}-{max(years)}" if years else "Not available"

    # Clustering rationale — truncate to avoid dominating the methodology prompt
    clustering_rationale = state.get("clustering_rationale") or ""
    if len(clustering_rationale) > _CLUSTERING_RATIONALE_MAX_LEN:
        clustering_rationale = clustering_rationale[:_CLUSTERING_RATIONALE_MAX_LEN] + "..."

    return TransparencyReport(
        # Discovery
        search_queries=list(dict.fromkeys(state.get("search_queries", []))),
        keyword_paper_count=keyword_paper_count,
        citation_paper_count=citation_paper_count,
        expert_paper_count=expert_paper_count,
        raw_results_count=state.get("raw_results_count") or 0,
        # Diffusion
        diffusion_stages=diffusion_stages,
        total_discovered=diffusion.get("total_papers_discovered", 0),
        total_rejected=diffusion.get("total_papers_rejected", 0),
        saturation_reason=diffusion.get("saturation_reason") or "unknown",
        # Quality filters
        min_citations_filter=quality_settings.get("min_citations_filter", 0),
        recency_years=quality_settings.get("recency_years_fallback", quality_settings.get("recency_years", 2)),
        recency_quota=quality_settings.get("recency_quota", 0.35),
        relevance_threshold=_RELEVANCE_THRESHOLD,
        # Processing
        papers_processed_count=len(state.get("papers_processed", [])),
        papers_failed_count=len(state.get("papers_failed", [])),
        metadata_only_count=len(state.get("metadata_only_dois", [])),
        fallback_substitutions_count=len(fallback_subs),
        fallback_substitutions_summary=fallback_summary,
        fallback_exhausted_count=len(state.get("fallback_exhausted", [])),
        # Clustering
        clustering_method=state.get("clustering_method") or "unknown",
        clustering_rationale=clustering_rationale,
        cluster_count=len(state.get("clusters", [])),
        # Corpus
        date_range=date_range,
        total_corpus_size=len(paper_summaries),
    )


def render_transparency_for_prompt(report: TransparencyReport) -> dict[str, str]:
    """Render TransparencyReport fields as template-ready strings.

    Pre-renders complex sub-sections into readable prose fragments so the
    LLM's job is prose quality, not data interpretation.

    Returns:
        Dict of string values keyed by template placeholder names.
    """
    # Search queries
    queries = report.get("search_queries", [])
    search_queries_formatted = "\n".join(f"- {q}" for q in queries) if queries else "No queries recorded"

    # Diffusion stages
    diffusion_stages_formatted = _format_diffusion_stages(report.get("diffusion_stages", []))

    # Saturation reason — pass through raw reason from the diffusion engine.
    # The engine already produces human-readable strings (e.g. "Reached maximum stages (3)").
    raw_reason = report.get("saturation_reason", "unknown")
    saturation_reason_formatted = raw_reason if raw_reason != "unknown" else "Not recorded"

    # Expert papers — conditionally include only if non-zero
    expert_count = report.get("expert_paper_count", 0)
    expert_papers_line = f"- Expert-identified papers: {expert_count} papers\n" if expert_count > 0 else ""

    # Full-text count = processed - metadata_only
    metadata_only_count = report.get("metadata_only_count", 0)
    papers_processed = report.get("papers_processed_count", 0)
    raw_full_text = papers_processed - metadata_only_count
    if raw_full_text < 0:
        logger.warning(
            "metadata_only_count (%d) exceeds papers_processed (%d); clamping full_text_count to 0",
            metadata_only_count, papers_processed,
        )
    full_text_count = max(0, raw_full_text)

    # Fallback note
    fallback_summary = report.get("fallback_substitutions_summary", "")
    exhausted = report.get("fallback_exhausted_count", 0)
    fallback_parts = []
    if fallback_summary:
        fallback_parts.append(fallback_summary)
    if exhausted:
        fallback_parts.append(f"{exhausted} additional papers could not be retrieved and had no viable fallback.")
    fallback_note = "\n".join(fallback_parts)

    return {
        "search_queries_formatted": _sanitise_for_template(search_queries_formatted),
        "keyword_paper_count": str(report.get("keyword_paper_count", 0)),
        "citation_paper_count": str(report.get("citation_paper_count", 0)),
        "expert_papers_line": expert_papers_line,
        "raw_results_count": str(report.get("raw_results_count", 0)),
        "relevance_threshold": str(report.get("relevance_threshold", 0.6)),
        "diffusion_stages_formatted": diffusion_stages_formatted,
        "saturation_reason_formatted": _sanitise_for_template(saturation_reason_formatted),
        "min_citations_filter": str(report.get("min_citations_filter", 0)),
        "recency_years": str(report.get("recency_years", 3)),
        "recency_quota_pct": str(int(report.get("recency_quota", 0.25) * 100)),
        "full_text_count": str(full_text_count),
        "metadata_only_count": str(metadata_only_count),
        "papers_failed_count": str(report.get("papers_failed_count", 0)),
        "fallback_note": _sanitise_for_template(fallback_note),
        "clustering_method": _sanitise_for_template(report.get("clustering_method", "unknown")),
        "clustering_rationale": _sanitise_for_template(report.get("clustering_rationale", "Not recorded")),
        "cluster_count": str(report.get("cluster_count", 0)),
        "date_range": report.get("date_range", "Not available"),
        "total_corpus_size": str(report.get("total_corpus_size", 0)),
    }


def _format_diffusion_stages(stages: list[DiffusionStageReport]) -> str:
    """Pre-render diffusion stages into readable sentence fragments."""
    if not stages:
        return "No citation expansion was performed."
    lines = []
    for s in stages:
        lines.append(
            f"Stage {s['stage_number']}: {s['forward_found']} forward citations, "
            f"{s['backward_found']} backward citations examined; "
            f"{s['new_relevant']} new relevant papers added, "
            f"{s['new_rejected']} rejected (coverage delta: {s['coverage_delta']:.2f})"
        )
    return "\n".join(lines)


def _format_fallback_summary(fallback_subs: list[dict]) -> str:
    """Format fallback substitutions as a readable summary for the prompt.

    Produces a count + reason breakdown rather than raw DOI lists.
    """
    if not fallback_subs:
        return ""

    reason_counts = Counter(sub.get("failure_reason", "unknown") for sub in fallback_subs)
    reason_parts = [f"{count} {reason.replace('_', ' ')}" for reason, count in reason_counts.items()]

    return (
        f"{len(fallback_subs)} papers were replaced with alternative papers "
        f"from the candidate pool due to retrieval failures ({', '.join(reason_parts)})."
    )
