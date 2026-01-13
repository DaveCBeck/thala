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
from typing import Literal

from workflows.web_research.state import (
    DeepResearchState,
    DiffusionState,
    TranslationConfig,
    parse_allocation,
)
from workflows.web_research.config.languages import get_language_config
from workflows.shared.workflow_state_store import save_workflow_state

from .construction import deep_research_graph
from .config import RECURSION_LIMITS

logger = logging.getLogger(__name__)


async def deep_research(
    query: str,
    depth: Literal["quick", "standard", "comprehensive"] = "standard",
    max_sources: int = 20,
    max_iterations: int = None,
    clarification_responses: dict[str, str] = None,
    # Language options
    language: str = None,
    translate_to: str = None,
    preserve_quotes: bool = True,
    # Researcher allocation
    researcher_allocation: str = None,
) -> dict:
    """
    Run deep research on a topic.

    This is the main entry point for the research workflow.

    Args:
        query: Research question or topic
        depth: Research depth
            - "quick": 2 iterations, ~5 min, focused questions
            - "standard": 4 iterations, ~15 min, balanced
            - "comprehensive": 8 iterations, 30+ min, exhaustive
        max_sources: Maximum web sources to consult
        max_iterations: Override default iteration count for depth
        clarification_responses: Pre-provided clarification answers
        language: Run workflow in this language (ISO 639-1 code, e.g., "es", "zh").
                  All prompts, queries, and output will be in target language.
                  Default: None (English)
        translate_to: After research, translate final report to this language.
                     Useful when researching in non-English but need English output.
        preserve_quotes: Keep direct quotes in original language when translating.
                        Default: True
        researcher_allocation: Number of parallel web researchers as single digit "1"-"3".
                              If None, supervisor LLM decides based on topic.

    Returns:
        DeepResearchState with:
        - final_report: Complete research report (in target language)
        - translated_report: If translate_to specified, the translated version
        - citations: List of sources used
        - store_record_id: UUID if saved to store
        - research_findings: All findings from researchers
        - errors: Any errors encountered

    Examples:
        # Standard English research
        result = await deep_research("Impact of AI on jobs in 2025")

        # Research in Spanish, output in Spanish
        result = await deep_research("impacto de IA en empleos", language="es")

        # Research in Spanish, translate output to English
        result = await deep_research(
            "impacto de IA en empleos",
            language="es",
            translate_to="en",
        )
    """
    # Generate a run_id for LangSmith tracing (allows inspection of runs)
    run_id = uuid.uuid4()

    # Determine primary language
    primary_lang = language or "en"
    primary_lang_config = get_language_config(primary_lang) if primary_lang != "en" else None

    # Build translation config if translate_to is specified
    translation_config = None
    if translate_to:
        translation_config = TranslationConfig(
            enabled=True,
            target_language=translate_to,
            preserve_quotes=preserve_quotes,
            preserve_citations=True,
        )

    # Parse researcher allocation if provided
    parsed_allocation = None
    if researcher_allocation:
        parsed_allocation = parse_allocation(researcher_allocation)

    initial_state: DeepResearchState = {
        "input": {
            "query": query,
            "depth": depth,
            "max_sources": max_sources,
            "max_iterations": max_iterations,
            "language": language,
            "translate_to": translate_to,
            "preserve_quotes": preserve_quotes,
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
        "researcher_allocation": parsed_allocation,  # User-specified or None (supervisor decides)
        "supervisor_messages": [],
        "diffusion": DiffusionState(
            iteration=0,
            max_iterations=max_iterations or {"quick": 2, "standard": 4, "comprehensive": 8}[depth],
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
        "translation_config": translation_config,
        "translated_report": None,
    }

    recursion_limit = RECURSION_LIMITS.get(depth, 100)
    lang_info = f", language={primary_lang}" if primary_lang != "en" else ""
    logger.info(f"Starting deep research: query='{query[:50]}...', depth={depth}{lang_info}, recursion_limit={recursion_limit}, run_id={run_id}")

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
            "translated_report": result.get("translated_report"),
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
