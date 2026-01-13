"""
Book query generation node.

Takes the outputs from web and academic research and generates a thematic
query suitable for the book finding workflow.
"""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm
from workflows.wrapped.state import WrappedResearchState

logger = logging.getLogger(__name__)


BOOK_QUERY_PROMPT = """You are helping prepare a thematic book search based on research findings.

## Original Query
{query}

## Web Research Summary
{web_summary}

## Academic Literature Summary
{academic_summary}

## Your Task
Based on these research findings, generate:

1. **Theme** (1-2 sentences): A thematic angle that would benefit from book exploration. Focus on the conceptual essence, not specific technical details. Books excel at providing deep perspective, narrative, and synthesis that complements web and academic sources.

2. **Brief** (2-3 sentences): Context and guidance for finding relevant books. What perspectives or insights from books would complement this research? Consider both non-fiction (for deeper analysis) and fiction (for experiential understanding).

Respond in this exact format:
THEME: <your theme here>
BRIEF: <your brief here>
"""

# TODO: Integrate memory context from Thala's coherence store to personalize recommendations
# This would allow the book query to account for what the user already knows and prefers.


async def generate_book_query(state: WrappedResearchState) -> dict[str, Any]:
    """Generate thematic query for book finding based on research results.

    Uses Sonnet to synthesize web and academic findings into a thematic
    angle suitable for book exploration.
    """
    web_result = state.get("web_result") or {}
    academic_result = state.get("academic_result") or {}

    # Handle cases where one or both failed
    web_output = web_result.get("final_output")
    academic_output = academic_result.get("final_output")

    web_summary = web_output if web_output else "No web research available."
    academic_summary = academic_output if academic_output else "No academic review available."

    # Truncate for prompt (take first ~4000 chars each to fit in context)
    max_chars = 4000
    if len(web_summary) > max_chars:
        web_summary = web_summary[:max_chars] + "\n\n[... truncated for brevity]"
    if len(academic_summary) > max_chars:
        academic_summary = academic_summary[:max_chars] + "\n\n[... truncated for brevity]"

    llm = get_llm(ModelTier.SONNET, max_tokens=1024)

    prompt = BOOK_QUERY_PROMPT.format(
        query=state["input"]["query"],
        web_summary=web_summary,
        academic_summary=academic_summary,
    )

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        content = response.content if isinstance(response.content, str) else str(response.content)

        # Parse response
        theme = ""
        brief = ""

        if "THEME:" in content:
            theme_start = content.index("THEME:") + 6
            theme_end = content.index("BRIEF:") if "BRIEF:" in content else len(content)
            theme = content[theme_start:theme_end].strip()

        if "BRIEF:" in content:
            brief_start = content.index("BRIEF:") + 6
            brief = content[brief_start:].strip()

        # Fallback if parsing fails
        if not theme:
            theme = state["input"]["query"]
            logger.warning("Failed to parse theme from LLM response, using original query as fallback")

        logger.debug(f"Generated book theme: {theme[:100]}...")

        return {
            "book_theme": theme,
            "book_brief": brief if brief else None,
            "current_phase": "book_query_generated",
        }

    except Exception as e:
        logger.error(f"Book query generation failed: {e}")
        return {
            "book_theme": state["input"]["query"],
            "book_brief": None,
            "current_phase": "book_query_generated",
            "errors": [{"phase": "generate_book_query", "error": str(e)}],
        }
