"""State initialization helpers for academic literature review workflow."""

import uuid
from datetime import datetime

from workflows.research.subgraphs.academic_lit_review.state import (
    AcademicLitReviewState,
    LitReviewInput,
    LitReviewDiffusionState,
)


def build_initial_state(
    input_data: LitReviewInput,
    quality_settings: dict,
) -> AcademicLitReviewState:
    """Build initial state for the literature review workflow."""
    return AcademicLitReviewState(
        input=input_data,
        quality_settings=quality_settings,
        keyword_papers=[],
        citation_papers=[],
        expert_papers=[],
        book_dois=[],
        diffusion=LitReviewDiffusionState(
            current_stage=0,
            max_stages=quality_settings["max_stages"],
            stages=[],
            saturation_threshold=quality_settings["saturation_threshold"],
            is_saturated=False,
            consecutive_low_coverage=0,
            total_papers_discovered=0,
            total_papers_relevant=0,
            total_papers_rejected=0,
        ),
        paper_corpus={},
        paper_summaries={},
        citation_edges=[],
        paper_nodes={},
        papers_to_process=[],
        papers_processed=[],
        papers_failed=[],
        bertopic_clusters=None,
        llm_topic_schema=None,
        clusters=[],
        section_drafts={},
        final_review=None,
        references=[],
        prisma_documentation=None,
        elasticsearch_ids={},
        zotero_keys={},
        started_at=datetime.utcnow(),
        completed_at=None,
        current_phase="discovery",
        current_status="Starting literature review",
        langsmith_run_id=str(uuid.uuid4()),
        errors=[],
    )
