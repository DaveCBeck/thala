"""
Final summary generation node.

Synthesizes outputs from all three research workflows into a
coherent combined summary.
"""

import logging
from datetime import datetime
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm
from workflows.wrapped.state import WrappedResearchState

logger = logging.getLogger(__name__)


SYNTHESIS_PROMPT = """You are synthesizing research from three complementary sources into a coherent summary.

## Original Query
{query}

## Web Research Findings
{web_summary}

## Academic Literature Review
{academic_summary}

## Book Recommendations
{book_summary}

## Your Task
Create a comprehensive summary (800-1200 words) that:

1. **Synthesizes Key Insights**: What are the most important takeaways across all three sources? Identify the core themes and findings that emerge from combining web research, academic literature, and book perspectives.

2. **Identifies Convergences**: Where do the web research, academic literature, and book recommendations agree or reinforce each other? These points of agreement often represent the most robust insights.

3. **Notes Unique Contributions**: What does each source contribute that the others don't?
   - Web research: Current trends, practical applications, recent developments
   - Academic literature: Empirical evidence, theoretical frameworks, methodological rigor
   - Books: Deep perspective, narrative synthesis, experiential understanding

4. **Highlights Tensions or Gaps**: Are there areas where sources disagree or where important questions remain unanswered?

5. **Suggests Next Steps**: Based on all findings, what further exploration would be valuable?

Write in clear, accessible prose. Use markdown formatting with headers for each section.
"""


async def generate_final_summary(state: WrappedResearchState) -> dict[str, Any]:
    """Generate LLM synthesis of all three research sources.

    Uses Opus with extended thinking for complex synthesis across
    web, academic, and book findings.
    """
    web_result = state.get("web_result") or {}
    academic_result = state.get("academic_result") or {}
    book_result = state.get("book_result") or {}

    # Get outputs, handling failures gracefully
    web_output = web_result.get("final_output")
    academic_output = academic_result.get("final_output")
    book_output = book_result.get("final_output")

    web_summary = web_output if web_output else "No web research available."
    academic_summary = academic_output if academic_output else "No academic review available."
    book_summary = book_output if book_output else "No book recommendations available."

    # Truncate each to fit in context (take first ~6000 chars each)
    max_chars = 6000
    if len(web_summary) > max_chars:
        web_summary = web_summary[:max_chars] + "\n\n[... truncated for brevity]"
    if len(academic_summary) > max_chars:
        academic_summary = academic_summary[:max_chars] + "\n\n[... truncated for brevity]"
    if len(book_summary) > max_chars:
        book_summary = book_summary[:max_chars] + "\n\n[... truncated for brevity]"

    # Use Opus for high-quality synthesis
    llm = get_llm(ModelTier.OPUS, max_tokens=4096)

    prompt = SYNTHESIS_PROMPT.format(
        query=state["input"]["query"],
        web_summary=web_summary,
        academic_summary=academic_summary,
        book_summary=book_summary,
    )

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        combined_summary = response.content if isinstance(response.content, str) else str(response.content)

        logger.info(f"Generated final summary: {len(combined_summary)} chars")

        return {
            "combined_summary": combined_summary,
            "current_phase": "summary_generated",
        }

    except Exception as e:
        logger.error(f"Failed to generate final summary: {e}")
        # Create a basic fallback summary
        fallback = f"""# Research Summary: {state['input']['query']}

## Overview
This research explored the topic through web sources, academic literature, and book recommendations.

## Status
- Web Research: {web_result.get('status', 'unknown')}
- Academic Review: {academic_result.get('status', 'unknown')}
- Book Finding: {book_result.get('status', 'unknown')}

## Note
Automated synthesis failed. Please review the individual outputs in top_of_mind for detailed findings.

Error: {e}
"""
        return {
            "combined_summary": fallback,
            "current_phase": "summary_generated",
            "errors": [{"phase": "generate_final_summary", "error": str(e)}],
        }
