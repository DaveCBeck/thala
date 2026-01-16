"""Phase 4d-e: Section writing workers and quality checking."""

import logging
from typing import Any

from langgraph.types import Send
from pydantic import BaseModel, Field

from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class SectionQuality(BaseModel):
    """Quality assessment for a section."""

    quality_score: float = Field(
        ge=0.0, le=1.0, description="Quality score from 0 to 1"
    )
    strengths: list[str] = Field(description="What the section does well")
    weaknesses: list[str] = Field(description="Areas for improvement")
    needs_revision: bool = Field(description="Whether section needs revision")


# =============================================================================
# Prompts
# =============================================================================


SECTION_WRITING_PROMPT = """Write a section for a synthesis document.

## Document Title
{document_title}

## Section: {section_title}
{section_description}

## Key Sources to Integrate
{key_sources}

## Available Research

### Academic Literature
{lit_review_excerpt}

### Web Research
{web_research_excerpt}

### Book Summaries
{book_summaries_excerpt}

## Writing Guidelines
1. Integrate insights from multiple sources
2. Use citations in [@ZOTKEY] format (e.g., [@SMITH2024])
3. Maintain academic tone while being accessible
4. Create clear transitions between ideas
5. Support claims with evidence from sources

Write the complete section content in markdown format. Do NOT include the section title as a header - just write the content."""


QUALITY_CHECK_PROMPT = """Assess the quality of this synthesis section.

## Section Title
{section_title}

## Section Description
{section_description}

## Section Content
{section_content}

## Evaluation Criteria
1. **Integration** (0.25): Does it synthesize multiple sources effectively?
2. **Citations** (0.25): Are claims properly supported with [@ZOTKEY] citations?
3. **Coherence** (0.25): Is the argument clear and well-structured?
4. **Depth** (0.25): Does it provide meaningful insights beyond surface-level?

Provide a quality score (0-1), list strengths and weaknesses, and indicate if revision is needed."""


# =============================================================================
# Nodes
# =============================================================================


def route_to_section_workers(state: dict) -> list[Send]:
    """Dispatch parallel workers to write each section.

    Creates Send() objects for each section in the synthesis structure,
    allowing them to be written in parallel.
    """
    synthesis_structure = state.get("synthesis_structure", {})
    sections = synthesis_structure.get("sections", [])

    if not sections:
        logger.warning("No sections to write")
        return [Send("assemble_sections", state)]

    logger.info(f"Dispatching {len(sections)} section writing workers")

    sends = []
    for section in sections:
        sends.append(
            Send(
                "write_section_worker",
                {
                    "section_id": section.get("section_id"),
                    "section_title": section.get("title"),
                    "section_description": section.get("description"),
                    "key_sources": section.get("key_sources", []),
                },
            )
        )

    return sends


async def write_section_worker(state: dict) -> dict[str, Any]:
    """Worker that writes a single section.

    Receives section info from Send(), writes the section content,
    and returns it for aggregation.
    """
    # Get section info from worker state
    section_id = state.get("section_id", "unknown")
    section_title = state.get("section_title", "Unknown Section")
    section_description = state.get("section_description", "")
    key_sources = state.get("key_sources", [])

    # Get context from parent state (passed through Send)
    input_data = state.get("input", {})
    quality_settings = state.get("quality_settings", {})
    synthesis_structure = state.get("synthesis_structure", {})
    lit_review_result = state.get("lit_review_result", {})
    supervision_result = state.get("supervision_result")
    web_research_results = state.get("web_research_results", [])
    book_summaries_cache = state.get("book_summaries_cache", {})

    document_title = synthesis_structure.get("title", input_data.get("topic", ""))

    logger.info(f"Writing section: {section_title}")

    try:
        # Build excerpts for context
        if supervision_result and supervision_result.get("final_report"):
            lit_review_excerpt = supervision_result["final_report"][:4000]
        elif lit_review_result and lit_review_result.get("final_report"):
            lit_review_excerpt = lit_review_result["final_report"][:4000]
        else:
            lit_review_excerpt = "No academic literature available."

        web_research_excerpt = "\n\n".join(
            f"**{r.get('query', 'Query')}**\n{r.get('final_report', '')[:1500]}"
            for r in web_research_results
            if r.get("status") == "success"
        )[:4000] or "No web research available."

        book_summaries_excerpt = "\n\n".join(
            f"**[@{zotkey}]**\n{summary[:2000]}"
            for zotkey, summary in book_summaries_cache.items()
        )[:4000] or "No book summaries available."

        key_sources_str = ", ".join(key_sources) if key_sources else "Use all available"

        # Use Opus if quality permits
        model_tier = (
            ModelTier.OPUS
            if quality_settings.get("use_opus_for_sections", True)
            else ModelTier.SONNET
        )
        llm = get_llm(model_tier, max_tokens=4000)

        prompt = SECTION_WRITING_PROMPT.format(
            document_title=document_title,
            section_title=section_title,
            section_description=section_description,
            key_sources=key_sources_str,
            lit_review_excerpt=lit_review_excerpt,
            web_research_excerpt=web_research_excerpt,
            book_summaries_excerpt=book_summaries_excerpt,
        )

        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        # Extract citations from content (simple regex)
        import re
        citations = re.findall(r'\[@([A-Za-z0-9_-]+)\]', content)

        section = {
            "section_id": section_id,
            "title": section_title,
            "content": content,
            "citations": list(set(citations)),
            "quality_score": None,
            "needs_revision": False,
        }

        logger.info(
            f"Wrote section '{section_title}': {len(content)} chars, "
            f"{len(citations)} citations"
        )

        return {
            "section_drafts": [section],
        }

    except Exception as e:
        logger.error(f"Section writing failed for '{section_title}': {e}")
        return {
            "section_drafts": [
                {
                    "section_id": section_id,
                    "title": section_title,
                    "content": f"[Section generation failed: {e}]",
                    "citations": [],
                    "quality_score": 0.0,
                    "needs_revision": True,
                }
            ],
            "errors": [{"phase": f"write_section_{section_id}", "error": str(e)}],
        }


async def check_section_quality(state: dict) -> dict[str, Any]:
    """Check quality of all written sections.

    Evaluates each section against quality criteria and marks
    those that need revision.
    """
    section_drafts = state.get("section_drafts", [])
    quality_settings = state.get("quality_settings", {})
    quality_threshold = quality_settings.get("section_quality_threshold", 0.7)

    if not section_drafts:
        logger.warning("No sections to check")
        return {"current_phase": "assemble"}

    logger.info(f"Phase 4e: Checking quality of {len(section_drafts)} sections")

    try:
        llm = get_llm(ModelTier.SONNET, max_tokens=1000)
        llm_structured = llm.with_structured_output(SectionQuality)

        updated_sections = []

        for section in section_drafts:
            try:
                prompt = QUALITY_CHECK_PROMPT.format(
                    section_title=section.get("title", "Unknown"),
                    section_description="",  # Would need from structure
                    section_content=section.get("content", "")[:5000],
                )

                result = await llm_structured.ainvoke([{"role": "user", "content": prompt}])

                section["quality_score"] = result.quality_score
                section["needs_revision"] = (
                    result.needs_revision or result.quality_score < quality_threshold
                )

                logger.debug(
                    f"Section '{section.get('title')}': "
                    f"score={result.quality_score:.2f}, needs_revision={section['needs_revision']}"
                )

            except Exception as e:
                logger.warning(f"Quality check failed for section: {e}")
                section["quality_score"] = 0.5
                section["needs_revision"] = False

            updated_sections.append(section)

        # Calculate overall quality
        scores = [s.get("quality_score", 0) for s in updated_sections if s.get("quality_score")]
        avg_score = sum(scores) / len(scores) if scores else 0
        needs_revision_count = sum(1 for s in updated_sections if s.get("needs_revision"))

        logger.info(
            f"Quality check complete: avg_score={avg_score:.2f}, "
            f"{needs_revision_count}/{len(updated_sections)} need revision"
        )

        return {
            "section_drafts": updated_sections,
            "current_phase": "assemble",
        }

    except Exception as e:
        logger.error(f"Quality checking failed: {e}")
        return {
            "current_phase": "assemble",
            "errors": [{"phase": "quality_check", "error": str(e)}],
        }


async def assemble_sections(state: dict) -> dict[str, Any]:
    """Assemble all sections into the final synthesis document."""
    section_drafts = state.get("section_drafts", [])
    synthesis_structure = state.get("synthesis_structure", {})
    input_data = state.get("input", {})

    document_title = synthesis_structure.get("title", input_data.get("topic", "Synthesis"))
    intro_guidance = synthesis_structure.get("introduction_guidance", "")
    conclusion_guidance = synthesis_structure.get("conclusion_guidance", "")

    logger.info(f"Assembling {len(section_drafts)} sections into final document")

    # Sort sections by section_id to maintain structure order
    structure_order = {
        s.get("section_id"): i
        for i, s in enumerate(synthesis_structure.get("sections", []))
    }
    sorted_sections = sorted(
        section_drafts,
        key=lambda s: structure_order.get(s.get("section_id"), 999)
    )

    # Build document
    lines = [
        f"# {document_title}",
        "",
    ]

    # Add introduction if we have guidance
    if intro_guidance:
        lines.extend([
            "## Introduction",
            "",
            f"*[Introduction based on: {intro_guidance}]*",
            "",
        ])

    # Add all sections
    for section in sorted_sections:
        lines.extend([
            f"## {section.get('title', 'Section')}",
            "",
            section.get("content", "[No content]"),
            "",
        ])

    # Add conclusion if we have guidance
    if conclusion_guidance:
        lines.extend([
            "## Conclusion",
            "",
            f"*[Conclusion based on: {conclusion_guidance}]*",
            "",
        ])

    final_report = "\n".join(lines)

    logger.info(f"Assembly complete: {len(final_report)} chars")

    return {
        "final_report": final_report,
        "current_phase": "editing",
    }
