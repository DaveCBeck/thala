"""Phase 3: Generate research targets for web research and book finding."""

import logging
from typing import Any

from langsmith import traceable
from pydantic import BaseModel, Field

from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)


class QueryGeneration(BaseModel):
    """Generated query for web research."""

    query: str = Field(description="Search query for web research")
    rationale: str = Field(description="Why this query is valuable")
    target_area: str = Field(description="What aspect of the topic this addresses")


class ThemeGeneration(BaseModel):
    """Generated theme for book finding."""

    theme: str = Field(description="Theme for book exploration")
    rationale: str = Field(description="Why this theme is valuable")
    book_angle: str = Field(
        description="Type of book to find: 'analogous', 'inspiring', or 'expressive'"
    )


class ResearchTargets(BaseModel):
    """Generated research targets."""

    queries: list[QueryGeneration] = Field(
        description="Queries for web research"
    )
    themes: list[ThemeGeneration] = Field(
        description="Themes for book finding"
    )


RESEARCH_TARGETS_PROMPT = """You are analyzing academic literature to identify gaps that need filling through web research and book exploration.

## Topic
{topic}

## Research Questions
{research_questions}

## Synthesis Brief
{synthesis_brief}

## Current Literature Review Summary
{lit_review_summary}

## Task
Based on the literature review above, generate:

1. **Web Research Queries** ({num_queries} queries):
   - Focus on recent developments, practical applications, industry perspectives
   - Target areas NOT well-covered by academic literature
   - Include queries for current statistics, case studies, practitioner insights

2. **Book Finding Themes** ({num_themes} themes):
   - Identify themes that would benefit from book-length exploration
   - Mix of:
     - **analogous**: Themes from different fields with transferable insights
     - **inspiring**: Themes that motivate action and change
     - **expressive**: Themes best captured through narrative/fiction
   - Focus on depth over breadth

Generate diverse, non-overlapping targets that complement the academic literature."""


@traceable(run_type="chain", name="SynthesisResearchTargets")
async def generate_research_targets(state: dict) -> dict[str, Any]:
    """Generate queries for web research and themes for book finding.

    Analyzes the literature review and supervision results to identify
    gaps that need filling through web research and book exploration.
    """
    input_data = state.get("input", {})
    quality_settings = state.get("quality_settings", {})
    lit_review_result = state.get("lit_review_result", {})
    supervision_result = state.get("supervision_result")

    topic = input_data.get("topic", "")
    research_questions = input_data.get("research_questions", [])
    synthesis_brief = input_data.get("synthesis_brief", "")

    # Use enhanced report if available, otherwise use original lit review
    if supervision_result and supervision_result.get("final_report"):
        lit_review_summary = supervision_result["final_report"][:10000]  # Truncate
    elif lit_review_result and lit_review_result.get("final_report"):
        lit_review_summary = lit_review_result["final_report"][:10000]
    else:
        lit_review_summary = "No literature review available."

    num_queries = quality_settings.get("web_research_runs", 3)
    num_themes = quality_settings.get("book_finding_runs", 3)

    logger.info(
        f"Phase 3: Generating {num_queries} queries and {num_themes} themes"
    )

    try:
        # Use Sonnet to generate research targets
        llm = get_llm(ModelTier.HAIKU, max_tokens=2000)
        llm_structured = llm.with_structured_output(ResearchTargets)

        prompt = RESEARCH_TARGETS_PROMPT.format(
            topic=topic,
            research_questions="\n".join(f"- {q}" for q in research_questions),
            synthesis_brief=synthesis_brief or "No specific angle provided.",
            lit_review_summary=lit_review_summary,
            num_queries=num_queries,
            num_themes=num_themes,
        )

        result = await llm_structured.ainvoke([{"role": "user", "content": prompt}])

        # Convert to state format
        generated_queries = [
            {
                "query": q.query,
                "rationale": q.rationale,
                "target_area": q.target_area,
            }
            for q in result.queries[:num_queries]
        ]

        generated_themes = [
            {
                "theme": t.theme,
                "rationale": t.rationale,
                "book_angle": t.book_angle,
            }
            for t in result.themes[:num_themes]
        ]

        logger.info(
            f"Phase 3 complete: generated {len(generated_queries)} queries, "
            f"{len(generated_themes)} themes"
        )

        return {
            "generated_queries": generated_queries,
            "generated_themes": generated_themes,
            "current_phase": "parallel_research",
        }

    except Exception as e:
        logger.error(f"Research target generation failed: {e}")
        # Return minimal defaults to continue workflow
        return {
            "generated_queries": [
                {"query": topic, "rationale": "Fallback query", "target_area": "general"}
            ],
            "generated_themes": [
                {"theme": topic, "rationale": "Fallback theme", "book_angle": "analogous"}
            ],
            "current_phase": "parallel_research",
            "errors": [{"phase": "research_targets", "error": str(e)}],
        }
