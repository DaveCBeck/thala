"""Main entry point for synthesis workflow.

Provides the public synthesis() function that orchestrates
the complete multi-phase synthesis workflow.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from workflows.shared.quality_config import QualityTier
from workflows.shared.tracing import (
    workflow_traceable,
    add_trace_metadata,
    merge_trace_config,
)
from workflows.shared.workflow_state_store import save_workflow_state
from workflows.wrappers.synthesis.state import SynthesisInput, SynthesisState
from workflows.wrappers.synthesis.quality_presets import SYNTHESIS_QUALITY_PRESETS
from .construction import synthesis_graph

logger = logging.getLogger(__name__)


@workflow_traceable(name="Synthesis", workflow_type="synthesis")
async def synthesis(
    topic: str,
    research_questions: list[str],
    synthesis_brief: Optional[str] = None,
    quality: QualityTier = "standard",
    language: str = "en",
    multi_lang_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Run complete synthesis workflow.

    Orchestrates multiple research workflows to create a comprehensive
    synthesized report:
    1. Academic literature review (via academic_lit_review)
    2. Supervision loops for theoretical depth (via enhance.supervision)
    3. Web research + book finding (parallel)
    4. Synthesis structure and section writing
    5. Editing (via enhance.editing)

    Args:
        topic: Research topic for synthesis
        research_questions: List of specific questions to address
        synthesis_brief: Optional description of desired synthesis angle
        quality: Quality tier
            - "test": Minimal testing (1 research iteration, simple synthesis)
            - "quick": Fast synthesis (2 iterations, skip supervision)
            - "standard": Balanced quality (3 iterations, full workflow)
            - "comprehensive": Thorough synthesis (4 iterations)
            - "high_quality": Maximum depth (5 iterations)
        language: ISO 639-1 language code (default: "en")
        multi_lang_config: Optional multi-language research configuration.
            When provided, research phases (lit review, web research, book finding)
            will use multi_lang_research() instead of direct workflow calls.
            Keys:
            - mode: "set_languages", "main_languages", or "all_languages"
            - languages: List of ISO 639-1 codes (required for set_languages mode)

    Returns:
        Dict containing:
        - final_report: Complete synthesis document
        - status: "success", "partial", or "failed"
        - langsmith_run_id: LangSmith tracing ID
        - errors: Any errors encountered
        - source_count: Number of sources used
        - started_at: Start timestamp
        - completed_at: End timestamp

    Example:
        result = await synthesis(
            topic="AI in healthcare",
            research_questions=[
                "How is AI being used in medical diagnosis?",
                "What are the ethical considerations?",
            ],
            synthesis_brief="Focus on practical applications in developing countries",
            quality="standard",
        )

        # With multi-language research
        result = await synthesis(
            topic="sustainable urban planning",
            research_questions=["How are cities adapting to climate change?"],
            quality="standard",
            multi_lang_config={
                "mode": "set_languages",
                "languages": ["en", "es", "de", "zh"],
            },
        )

        # Access results
        print(f"Report length: {len(result['final_report'])} chars")

        # Save to file
        with open("synthesis.md", "w") as f:
            f.write(result["final_report"])
    """
    # Get quality settings
    if quality not in SYNTHESIS_QUALITY_PRESETS:
        logger.warning(f"Unknown quality '{quality}', using 'standard'")
        quality = "standard"

    quality_settings = dict(SYNTHESIS_QUALITY_PRESETS[quality])

    # Add dynamic trace metadata for LangSmith filtering
    add_trace_metadata({
        "quality_tier": quality,
        "language": language,
        "topic": topic[:50],
    })

    # Build input
    input_data = SynthesisInput(
        topic=topic,
        research_questions=research_questions,
        synthesis_brief=synthesis_brief,
        quality=quality,
        language_code=language,
        multi_lang_config=multi_lang_config,
    )

    # Initialize state
    langsmith_run_id = str(uuid.uuid4())
    initial_state = SynthesisState(
        input=input_data,
        quality_settings=quality_settings,
        # Phase 1: Literature Review
        lit_review_result=None,
        paper_corpus={},
        paper_summaries={},
        zotero_keys={},
        # Phase 2: Supervision
        supervision_result=None,
        # Phase 3: Research Targets
        generated_queries=[],
        generated_themes=[],
        # Phase 3b: Parallel Research
        web_research_results=[],
        book_finding_results=[],
        # Phase 4: Synthesis
        synthesis_structure=None,
        selected_books=[],
        book_summaries_cache={},
        section_drafts=[],
        # Phase 5: Editing
        editing_result=None,
        # Output
        final_report=None,
        final_report_with_references=None,
        # Metadata
        started_at=datetime.utcnow(),
        completed_at=None,
        current_phase="starting",
        status=None,
        langsmith_run_id=langsmith_run_id,
        errors=[],
    )

    multi_lang_msg = ""
    if multi_lang_config is not None:
        multi_lang_msg = f", multi_lang_mode={multi_lang_config.get('mode', 'set_languages')}"

    logger.info(
        f"Starting synthesis workflow: '{topic}' "
        f"(quality={quality}, questions={len(research_questions)}, language={language}{multi_lang_msg})"
    )
    logger.debug(f"LangSmith run ID: {langsmith_run_id}")

    try:
        result = await synthesis_graph.ainvoke(
            initial_state,
            config=merge_trace_config({
                "run_name": f"synthesis:{topic[:30]}",
                "recursion_limit": 200,  # Higher limit for many parallel workers
            }),
        )

        final_report = result.get("final_report", "")
        errors = result.get("errors", [])
        status = result.get("status", "failed")

        # Calculate source count
        web_sources = sum(
            r.get("source_count", 0)
            for r in result.get("web_research_results", [])
        )
        books_processed = sum(
            len(r.get("processed_books", []))
            for r in result.get("book_finding_results", [])
        )
        papers_analyzed = len(result.get("paper_corpus", {}))
        source_count = web_sources + books_processed + papers_analyzed

        # Save full state for downstream workflows
        save_workflow_state(
            workflow_name="synthesis",
            run_id=langsmith_run_id,
            state={
                "input": dict(input_data),
                "paper_corpus": result.get("paper_corpus", {}),
                "paper_summaries": result.get("paper_summaries", {}),
                "zotero_keys": result.get("zotero_keys", {}),
                "web_research_results": result.get("web_research_results", []),
                "book_finding_results": result.get("book_finding_results", []),
                "synthesis_structure": result.get("synthesis_structure"),
                "selected_books": result.get("selected_books", []),
                "section_drafts": result.get("section_drafts", []),
                "final_report": final_report,
                "started_at": initial_state["started_at"],
                "completed_at": result.get("completed_at"),
            },
        )

        logger.info(
            f"Synthesis workflow completed: status={status}, "
            f"report_length={len(final_report)}, "
            f"sources={source_count}"
        )

        return {
            "final_report": final_report,
            "status": status,
            "langsmith_run_id": langsmith_run_id,
            "errors": errors,
            "source_count": source_count,
            "started_at": initial_state["started_at"],
            "completed_at": result.get("completed_at"),
        }

    except Exception as e:
        logger.error(f"Synthesis workflow failed: {e}", exc_info=True)
        return {
            "final_report": f"# Synthesis Failed\n\nError: {e}",
            "status": "failed",
            "langsmith_run_id": langsmith_run_id,
            "errors": [{"phase": "unknown", "error": str(e)}],
            "source_count": 0,
            "started_at": initial_state["started_at"],
            "completed_at": datetime.utcnow(),
        }
