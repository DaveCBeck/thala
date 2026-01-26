"""Deep-dive writing node for evening_reads workflow.

Writes individual deep-dive articles with distinctiveness enforcement.
Each deep-dive gets a "must avoid" list of themes covered by other deep-dives.
"""

import logging
import re
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic

from workflows.shared.llm_utils import ModelTier

from ..prompts import (
    DEEP_DIVE_PUZZLE_PROMPT_FULL,
    DEEP_DIVE_FINDING_PROMPT_FULL,
    DEEP_DIVE_CONTRARIAN_PROMPT_FULL,
    DEEP_DIVE_USER_TEMPLATE,
)
from ..state import DeepDiveDraft, EnrichedContent

# Map structural approach to prompt
STRUCTURAL_PROMPTS = {
    "puzzle": DEEP_DIVE_PUZZLE_PROMPT_FULL,
    "finding": DEEP_DIVE_FINDING_PROMPT_FULL,
    "contrarian": DEEP_DIVE_CONTRARIAN_PROMPT_FULL,
}

logger = logging.getLogger(__name__)

# Target 2500-3500 words = ~10000-14000 tokens output
MAX_TOKENS = 14000

CITATION_PATTERN = r"\[@([^\]]+)\]"


def _extract_citations(text: str) -> list[str]:
    """Extract all [@KEY] citations from text."""
    matches = re.findall(CITATION_PATTERN, text)
    keys = set()
    for match in matches:
        for key in match.split(";"):
            key = key.strip().lstrip("@")
            if key:
                keys.add(key)
    return sorted(keys)


def _extract_relevant_sections(
    lit_review: str, section_names: list[str], max_chars: int = 20000
) -> str:
    """Extract sections from the literature review that match the given names.

    Falls back to returning a truncated version if no sections match.
    """
    if not section_names:
        # No sections specified, return truncated review
        result = lit_review[:max_chars] if len(lit_review) > max_chars else lit_review
        if len(lit_review) > max_chars:
            logger.warning(
                f"No sections specified; truncated lit review from {len(lit_review)} to {max_chars} chars"
            )
        return result

    # Try to find markdown headers matching the section names
    lines = lit_review.split("\n")
    relevant_lines = []
    matched_headers = []
    in_relevant_section = False
    current_header_level = 0

    for line in lines:
        # Check if this is a header
        header_match = re.match(r"^(#+)\s+(.+)$", line)
        if header_match:
            level = len(header_match.group(1))
            header_text = header_match.group(2).strip()

            # Check if this header matches any of our section names
            matches_section = any(
                section.lower() in header_text.lower() for section in section_names
            )

            if matches_section:
                in_relevant_section = True
                current_header_level = level
                matched_headers.append(header_text)
                relevant_lines.append(line)
            elif in_relevant_section and level <= current_header_level:
                # We've moved to a sibling or parent section
                in_relevant_section = False
            elif in_relevant_section:
                relevant_lines.append(line)
        elif in_relevant_section:
            relevant_lines.append(line)

    if relevant_lines:
        excerpt = "\n".join(relevant_lines)
        original_len = len(excerpt)
        if original_len > max_chars:
            logger.warning(
                f"Truncating lit review excerpt from {original_len} to {max_chars} chars"
            )
            excerpt = excerpt[:max_chars] + "\n\n[... truncated ...]"
        logger.info(
            f"Extracted {len(excerpt)} chars from {len(matched_headers)} sections: {matched_headers}"
        )
        return excerpt

    # Fallback: return truncated full review
    result = lit_review[:max_chars] if len(lit_review) > max_chars else lit_review
    if len(lit_review) > max_chars:
        logger.warning(
            f"No matching sections found; truncated full review from {len(lit_review)} to {max_chars} chars"
        )
    else:
        logger.info(
            f"No matching sections found; returning full review ({len(lit_review)} chars)"
        )
    return result


async def write_deep_dive_node(state: dict) -> dict[str, Any]:
    """Write a single deep-dive article.

    This node is called via Send() with the assignment details and content.

    Expected state keys from Send():
        - deep_dive_id: Which deep-dive this is
        - title: Article title
        - theme: Article theme description
        - anchor_keys: Zotero keys for this deep-dive
        - relevant_sections: Section names to focus on
        - must_avoid: Themes covered by other deep-dives (for distinctiveness)
        - enriched_content: List of EnrichedContent for this deep-dive
        - literature_review: Full literature review for context

    Returns:
        State update with deep_dive_drafts list (aggregated via add reducer)
    """
    deep_dive_id: Literal["deep_dive_1", "deep_dive_2", "deep_dive_3"] = state.get(
        "deep_dive_id"
    )
    title = state.get("title", "Untitled")
    theme = state.get("theme", "")
    structural_approach = state.get("structural_approach", "puzzle")
    relevant_sections = state.get("relevant_sections", [])
    must_avoid = state.get("must_avoid", [])
    enriched_content: list[EnrichedContent] = state.get("enriched_content", [])
    lit_review = state.get("literature_review", "")

    if not deep_dive_id:
        return {
            "errors": [{"node": "write_deep_dive", "error": "Missing deep_dive_id"}]
        }

    # Build source content from enriched content for this deep-dive
    source_parts = []
    for ec in enriched_content:
        if ec["deep_dive_id"] == deep_dive_id:
            source_parts.append(
                f"## Source: {ec['zotero_key']} ({ec['content_level']})\n\n"
                f"{ec['content']}"
            )

    source_content = (
        "\n\n---\n\n".join(source_parts)
        if source_parts
        else "[No source content available]"
    )

    # Extract relevant sections from the literature review
    lit_review_excerpt = _extract_relevant_sections(lit_review, relevant_sections)

    # Build must_avoid string
    must_avoid_str = (
        "\n".join(f"- {item}" for item in must_avoid)
        if must_avoid
        else "None specified"
    )

    # Select prompt based on structural approach
    prompt_template = STRUCTURAL_PROMPTS.get(
        structural_approach, STRUCTURAL_PROMPTS["puzzle"]
    )

    # Format system prompt with assignment details
    system_prompt = prompt_template.format(
        title=title,
        theme=theme,
        must_avoid=must_avoid_str,
    )

    user_prompt = DEEP_DIVE_USER_TEMPLATE.format(
        source_content=source_content,
        literature_review_excerpt=lit_review_excerpt,
    )

    logger.info(
        f"Writing deep-dive {deep_dive_id}: '{title}' "
        f"(approach={structural_approach}, {len(source_parts)} sources, {len(must_avoid)} themes to avoid)"
    )

    try:
        llm = ChatAnthropic(
            model=ModelTier.OPUS.value,
            max_tokens=MAX_TOKENS,
        )

        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        word_count = len(content.split())
        citation_keys = _extract_citations(content)

        logger.info(
            f"Generated deep-dive {deep_dive_id}: {word_count} words, "
            f"{len(citation_keys)} citations"
        )

        draft = DeepDiveDraft(
            id=deep_dive_id,
            title=title,
            content=content,
            word_count=word_count,
            citation_keys=citation_keys,
        )

        return {"deep_dive_drafts": [draft]}

    except Exception as e:
        logger.error(f"Failed to write deep-dive {deep_dive_id}: {e}")
        return {
            "errors": [
                {"node": f"write_deep_dive_{deep_dive_id}", "error": str(e)}
            ]
        }
