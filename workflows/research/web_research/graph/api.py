"""
Main deep research workflow graph.

Implements the Self-Balancing Diffusion Algorithm with:
- Memory-first research (search Thala stores before web)
- Plan customization based on existing knowledge
- Parallel researcher agents (up to 3 concurrent)
- Iterative refinement with completeness checking
- Final report with citations saved to store

Flow:
1. clarify_intent - Ask clarifying questions if needed
2. create_brief - Generate structured research brief
3. search_memory - Search Thala stores for existing knowledge
4. iterate_plan - Customize plan based on memory (OPUS)
5. supervisor - Coordinate research with diffusion algorithm (OPUS)
   - Loop: generate questions -> researchers -> refine draft
6. final_report - Generate comprehensive report (OPUS)
7. save_findings - Save to store
"""

from core.config import configure_langsmith

configure_langsmith()

import logging
import uuid
from datetime import datetime

from workflows.shared.quality_config import QualityTier
from workflows.research.web_research.state import (
    DeepResearchState,
    DiffusionState,
)
from workflows.research.web_research.config.languages import get_language_config
from workflows.shared.workflow_state_store import save_workflow_state

from .construction import deep_research_graph
from .config import RECURSION_LIMITS

logger = logging.getLogger(__name__)


async def deep_research(
    query: str,
    quality: QualityTier = "standard",
    max_iterations: int = None,
    clarification_responses: dict[str, str] = None,
    language: str = None,
) -> dict:
    """
    Run deep research on a topic.

    This is the main entry point for the research workflow.

    Args:
        query: Research question or topic
        quality: Research quality tier
            - "test": 1 iteration, ~1 min, minimal testing
            - "quick": 2 iterations, ~5 min, focused questions
            - "standard": 4 iterations, ~15 min, balanced
            - "comprehensive": 8 iterations, 30+ min, exhaustive
            - "high_quality": 12 iterations, 45+ min, maximum depth
        max_iterations: Override default iteration count for quality tier
        clarification_responses: Pre-provided clarification answers
        language: Run workflow in this language (ISO 639-1 code, e.g., "es", "zh").
                  All prompts, queries, and output will be in target language.
                  Default: None (English)

    Returns:
        Dict with:
        - final_report: Complete research report (in target language)
        - status: "success", "partial", or "failed"
        - langsmith_run_id: LangSmith tracing ID
        - errors: Any errors encountered
        - source_count: Number of sources used
        - started_at: Workflow start time
        - completed_at: Workflow end time

    Examples:
        # Standard English research
        result = await deep_research("Impact of AI on jobs in 2025")

        # Research in Spanish
        result = await deep_research("impacto de IA en empleos", language="es")
    """
    # Generate a run_id for LangSmith tracing (allows inspection of runs)
    run_id = uuid.uuid4()

    # Determine primary language
    primary_lang = language or "en"
    primary_lang_config = get_language_config(primary_lang) if primary_lang != "en" else None

    initial_state: DeepResearchState = {
        "input": {
            "query": query,
            "quality": quality,
            "max_iterations": max_iterations,
            "language": language,
        },
        "clarification_needed": False,
        "clarification_questions": [],
        "clarification_responses": clarification_responses,
        "research_brief": None,
        "memory_findings": [],
        "memory_context": "",
        "research_plan": None,
        "pending_questions": [],
        "active_researchers": 0,
        "research_findings": [],
        "supervisor_messages": [],
        "diffusion": DiffusionState(
            iteration=0,
            max_iterations=max_iterations or {
                "test": 1,
                "quick": 2,
                "standard": 4,
                "comprehensive": 8,
                "high_quality": 12,
            }[quality],
            completeness_score=0.0,
            areas_explored=[],
            areas_to_explore=[],
        ),
        "draft_report": None,
        "final_report": None,
        "citations": [],
        "citation_keys": [],
        "store_record_id": None,
        "zotero_key": None,
        "errors": [],
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "current_status": "starting",
        "status": None,  # Will be set on completion
        "langsmith_run_id": str(run_id),

        # Language support
        "primary_language": primary_lang,
        "primary_language_config": primary_lang_config,
    }

    recursion_limit = RECURSION_LIMITS.get(quality, 100)
    lang_info = f", language={primary_lang}" if primary_lang != "en" else ""
    logger.info(f"Starting deep research: query='{query[:50]}...', quality={quality}{lang_info}, recursion_limit={recursion_limit}, run_id={run_id}")

    result = await deep_research_graph.ainvoke(
        initial_state,
        config={
            "recursion_limit": recursion_limit,
            "run_id": run_id,
            "run_name": f"deep_research:{query[:30]}",
        },
    )

    logger.info(
        f"Deep research complete: status={result.get('current_status')}, "
        f"findings={len(result.get('research_findings', []))}, "
        f"iterations={result.get('diffusion', {}).get('iteration', 0)}"
    )

    # Determine standardized status
    errors = result.get("errors", [])
    final_report = result.get("final_report")
    if final_report and not errors:
        status = "success"
    elif final_report and errors:
        status = "partial"
    else:
        status = "failed"

    # Save full state for downstream workflows (in dev/test mode)
    save_workflow_state(
        workflow_name="web_research",
        run_id=str(run_id),
        state={
            "input": initial_state.get("input"),
            "research_findings": result.get("research_findings", []),
            "research_brief": result.get("research_brief"),
            "memory_findings": result.get("memory_findings", []),
            "memory_context": result.get("memory_context", ""),
            "citations": result.get("citations", []),
            "diffusion": result.get("diffusion", {}),
            "draft_report": result.get("draft_report"),
            "final_report": final_report,
            "store_record_id": result.get("store_record_id"),
            "started_at": initial_state.get("started_at"),
            "completed_at": result.get("completed_at"),
        },
    )

    return {
        "final_report": final_report,
        "status": status,
        "langsmith_run_id": str(run_id),
        "errors": errors,
        "source_count": len(result.get("research_findings", [])),
        "started_at": initial_state.get("started_at"),
        "completed_at": result.get("completed_at"),
    }
