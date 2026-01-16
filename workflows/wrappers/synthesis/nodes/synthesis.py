"""Phase 4: Synthesis nodes for structure suggestion and book selection."""

import logging
from typing import Any

from pydantic import BaseModel, Field

from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class SectionSuggestion(BaseModel):
    """A suggested section for the synthesis."""

    section_id: str = Field(description="Unique identifier for section")
    title: str = Field(description="Section title")
    description: str = Field(description="What this section should cover")
    key_sources: list[str] = Field(
        description="Key sources to integrate (DOIs or zotero keys)"
    )


class StructureSuggestion(BaseModel):
    """Suggested structure for the synthesis document."""

    title: str = Field(description="Document title")
    sections: list[SectionSuggestion] = Field(description="Suggested sections")
    introduction_guidance: str = Field(description="Guidance for introduction")
    conclusion_guidance: str = Field(description="Guidance for conclusion")


class BookSelection(BaseModel):
    """A selected book for deep integration."""

    zotero_key: str = Field(description="Zotero citation key")
    title: str = Field(description="Book title")
    rationale: str = Field(description="Why this book should be deeply integrated")


class BookSelections(BaseModel):
    """Selected books for synthesis."""

    books: list[BookSelection] = Field(description="Selected books")


# =============================================================================
# Prompts
# =============================================================================


STRUCTURE_PROMPT = """You are designing the structure for a comprehensive synthesis document.

## Topic
{topic}

## Research Questions
{research_questions}

## Synthesis Brief
{synthesis_brief}

## Available Sources

### Academic Literature Review
{lit_review_summary}

### Web Research Findings
{web_research_summary}

### Book Recommendations
{book_summary}

## Task
Design a synthesis structure that:
1. Integrates insights from academic literature, web research, and books
2. Addresses all research questions
3. Has clear logical flow
4. Identifies which sources should be cited in each section

Provide a title, 4-7 sections with descriptions, and guidance for introduction and conclusion."""


SIMPLE_SYNTHESIS_PROMPT = """Create a brief synthesis based on the following research.

## Topic
{topic}

## Research Questions
{research_questions}

## Academic Literature
{lit_review_summary}

## Web Research
{web_research_summary}

## Book Insights
{book_summary}

Write a comprehensive synthesis that:
1. Integrates all sources
2. Addresses research questions
3. Uses citations in [@ZOTKEY] format where available
4. Is well-structured with clear sections

Output the complete synthesis document in markdown format."""


BOOK_SELECTION_PROMPT = """Select the most valuable books for deep integration into the synthesis.

## Topic
{topic}

## Synthesis Brief
{synthesis_brief}

## Available Books
{book_list}

## Task
Select up to {max_books} books that would provide the deepest value for synthesis.
Consider:
- Relevance to the topic and research questions
- Unique perspectives not covered by academic literature
- Depth of insight available

For each selected book, explain why it should be deeply integrated."""


# =============================================================================
# Nodes
# =============================================================================


async def suggest_structure(state: dict) -> dict[str, Any]:
    """Suggest synthesis structure based on all gathered research.

    Uses Opus (if quality permits) to design a comprehensive structure
    that integrates academic literature, web research, and book insights.
    """
    input_data = state.get("input", {})
    quality_settings = state.get("quality_settings", {})
    lit_review_result = state.get("lit_review_result", {})
    supervision_result = state.get("supervision_result")
    web_research_results = state.get("web_research_results", [])
    book_finding_results = state.get("book_finding_results", [])

    topic = input_data.get("topic", "")
    research_questions = input_data.get("research_questions", [])
    synthesis_brief = input_data.get("synthesis_brief", "")

    # Build source summaries
    if supervision_result and supervision_result.get("final_report"):
        lit_review_summary = supervision_result["final_report"][:8000]
    elif lit_review_result and lit_review_result.get("final_report"):
        lit_review_summary = lit_review_result["final_report"][:8000]
    else:
        lit_review_summary = "No academic literature available."

    web_research_summary = "\n\n".join(
        f"### Query: {r.get('query', 'Unknown')}\n{r.get('final_report', '')[:2000]}"
        for r in web_research_results
        if r.get("status") == "success"
    )[:8000] or "No web research available."

    book_summary = "\n\n".join(
        f"### Theme: {r.get('theme', 'Unknown')}\n{r.get('final_report', '')[:2000]}"
        for r in book_finding_results
        if r.get("status") == "success"
    )[:8000] or "No book insights available."

    logger.info("Phase 4a: Suggesting synthesis structure")

    try:
        # Use Opus if quality permits
        model_tier = (
            ModelTier.OPUS
            if quality_settings.get("use_opus_for_structure", True)
            else ModelTier.SONNET
        )
        llm = get_llm(model_tier, max_tokens=4000)
        llm_structured = llm.with_structured_output(StructureSuggestion)

        prompt = STRUCTURE_PROMPT.format(
            topic=topic,
            research_questions="\n".join(f"- {q}" for q in research_questions),
            synthesis_brief=synthesis_brief or "No specific angle provided.",
            lit_review_summary=lit_review_summary,
            web_research_summary=web_research_summary,
            book_summary=book_summary,
        )

        result = await llm_structured.ainvoke([{"role": "user", "content": prompt}])

        # Convert to state format
        synthesis_structure = {
            "title": result.title,
            "sections": [
                {
                    "section_id": s.section_id,
                    "title": s.title,
                    "description": s.description,
                    "key_sources": s.key_sources,
                }
                for s in result.sections
            ],
            "introduction_guidance": result.introduction_guidance,
            "conclusion_guidance": result.conclusion_guidance,
        }

        logger.info(
            f"Structure suggested: '{result.title}' with {len(result.sections)} sections"
        )

        return {
            "synthesis_structure": synthesis_structure,
            "current_phase": "select_books",
        }

    except Exception as e:
        logger.error(f"Structure suggestion failed: {e}")
        # Return minimal structure
        return {
            "synthesis_structure": {
                "title": f"Synthesis: {topic}",
                "sections": [
                    {
                        "section_id": "main",
                        "title": "Main Findings",
                        "description": "Synthesize all findings",
                        "key_sources": [],
                    }
                ],
                "introduction_guidance": "Introduce the topic and research questions.",
                "conclusion_guidance": "Summarize key insights and implications.",
            },
            "current_phase": "select_books",
            "errors": [{"phase": "suggest_structure", "error": str(e)}],
        }


async def simple_synthesis(state: dict) -> dict[str, Any]:
    """Create a simple synthesis without full structure (for test mode).

    Uses Sonnet to create a quick synthesis that integrates all sources
    without the full section-by-section writing process.
    """
    input_data = state.get("input", {})
    lit_review_result = state.get("lit_review_result", {})
    supervision_result = state.get("supervision_result")
    web_research_results = state.get("web_research_results", [])
    book_finding_results = state.get("book_finding_results", [])

    topic = input_data.get("topic", "")
    research_questions = input_data.get("research_questions", [])

    # Build source summaries
    if supervision_result and supervision_result.get("final_report"):
        lit_review_summary = supervision_result["final_report"][:10000]
    elif lit_review_result and lit_review_result.get("final_report"):
        lit_review_summary = lit_review_result["final_report"][:10000]
    else:
        lit_review_summary = "No academic literature available."

    web_research_summary = "\n\n".join(
        f"### {r.get('query', 'Query')}\n{r.get('final_report', '')[:3000]}"
        for r in web_research_results
        if r.get("status") == "success"
    )[:10000] or "No web research available."

    book_summary = "\n\n".join(
        f"### {r.get('theme', 'Theme')}\n{r.get('final_report', '')[:3000]}"
        for r in book_finding_results
        if r.get("status") == "success"
    )[:10000] or "No book insights available."

    logger.info("Creating simple synthesis (test mode)")

    try:
        llm = get_llm(ModelTier.SONNET, max_tokens=8000)

        prompt = SIMPLE_SYNTHESIS_PROMPT.format(
            topic=topic,
            research_questions="\n".join(f"- {q}" for q in research_questions),
            lit_review_summary=lit_review_summary,
            web_research_summary=web_research_summary,
            book_summary=book_summary,
        )

        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        synthesis = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        logger.info(f"Simple synthesis complete: {len(synthesis)} chars")

        return {
            "final_report": synthesis,
            "current_phase": "editing",
        }

    except Exception as e:
        logger.error(f"Simple synthesis failed: {e}")
        return {
            "final_report": f"# {topic}\n\nSynthesis generation failed: {e}",
            "current_phase": "editing",
            "errors": [{"phase": "simple_synthesis", "error": str(e)}],
        }


async def select_books(state: dict) -> dict[str, Any]:
    """Select books for deep integration in synthesis.

    Analyzes available books and selects the most valuable ones
    for detailed integration (fetching their 10x summaries).
    """
    input_data = state.get("input", {})
    quality_settings = state.get("quality_settings", {})
    book_finding_results = state.get("book_finding_results", [])

    topic = input_data.get("topic", "")
    synthesis_brief = input_data.get("synthesis_brief", "")
    max_books = quality_settings.get("max_books_to_select", 4)

    # Collect all books with zotero keys
    all_books = []
    for result in book_finding_results:
        if result.get("status") == "success":
            for book in result.get("processed_books", []):
                if book.get("zotero_key"):
                    all_books.append(book)

    if not all_books:
        logger.info("No books with zotero keys available for selection")
        return {
            "selected_books": [],
            "current_phase": "fetch_book_summaries",
        }

    logger.info(f"Phase 4b: Selecting up to {max_books} from {len(all_books)} books")

    try:
        llm = get_llm(ModelTier.SONNET, max_tokens=2000)
        llm_structured = llm.with_structured_output(BookSelections)

        book_list = "\n".join(
            f"- **{b.get('title', 'Unknown')}** [@{b.get('zotero_key')}]\n"
            f"  Authors: {b.get('authors', 'Unknown')}\n"
            f"  Summary: {(b.get('content_summary') or 'No summary')[:300]}..."
            for b in all_books
        )

        prompt = BOOK_SELECTION_PROMPT.format(
            topic=topic,
            synthesis_brief=synthesis_brief or "No specific angle provided.",
            book_list=book_list,
            max_books=max_books,
        )

        result = await llm_structured.ainvoke([{"role": "user", "content": prompt}])

        selected_books = [
            {
                "zotero_key": b.zotero_key,
                "title": b.title,
                "rationale": b.rationale,
            }
            for b in result.books[:max_books]
        ]

        logger.info(f"Selected {len(selected_books)} books for deep integration")

        return {
            "selected_books": selected_books,
            "current_phase": "fetch_book_summaries",
        }

    except Exception as e:
        logger.error(f"Book selection failed: {e}")
        # Fall back to selecting first N books
        selected_books = [
            {
                "zotero_key": b.get("zotero_key"),
                "title": b.get("title", "Unknown"),
                "rationale": "Fallback selection",
            }
            for b in all_books[:max_books]
            if b.get("zotero_key")
        ]

        return {
            "selected_books": selected_books,
            "current_phase": "fetch_book_summaries",
            "errors": [{"phase": "select_books", "error": str(e)}],
        }
