"""Overview writing node for evening_reads workflow.

Writes the overview article that synthesizes the big picture and references
the deep-dives without duplicating their content.
"""

import logging
import re
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm

from ..prompts import EDITORIAL_STANCE_SECTION, OVERVIEW_SYSTEM_PROMPT_FULL, OVERVIEW_USER_TEMPLATE
from ..state import DeepDiveDraft, OverviewDraft, EveningReadsState

logger = logging.getLogger(__name__)

# Target 2000-3000 words = ~8000-12000 tokens output
MAX_TOKENS = 12000

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


def _format_deep_dive_summaries(drafts: list[DeepDiveDraft]) -> str:
    """Format deep-dive information for the system prompt."""
    if not drafts:
        return "No deep-dives available."

    summaries = []
    for draft in drafts:
        # Extract first paragraph as a summary hint
        first_para = draft["content"].split("\n\n")[0][:300] if draft["content"] else ""
        summaries.append(
            f"**{draft['id'].replace('_', ' ').title()}**: \"{draft['title']}\"\n"
            f"Theme preview: {first_para}..."
        )

    return "\n\n".join(summaries)


def _format_deep_dive_list(drafts: list[DeepDiveDraft]) -> str:
    """Format deep-dive list for the user prompt."""
    if not drafts:
        return "No deep-dives planned."

    items = []
    for i, draft in enumerate(drafts, 1):
        items.append(f"{i}. \"{draft['title']}\" ({draft['id']})")

    return "\n".join(items)


async def write_overview_node(state: EveningReadsState) -> dict[str, Any]:
    """Write the overview article.

    This node runs after all deep-dives are complete and references them.

    Returns:
        State update with overview_draft
    """
    lit_review = state["input"]["literature_review"]
    editorial_stance = state["input"].get("editorial_stance", "")
    deep_dive_drafts = state.get("deep_dive_drafts", [])
    overview_scope = state.get("overview_scope", "")

    if not deep_dive_drafts:
        logger.warning("No deep-dive drafts available for overview")
        return {
            "errors": [
                {"node": "write_overview", "error": "No deep-dive drafts available"}
            ]
        }

    # Format deep-dive information for prompts
    deep_dive_summaries = _format_deep_dive_summaries(deep_dive_drafts)
    deep_dive_list = _format_deep_dive_list(deep_dive_drafts)

    # Build system prompt with deep-dive information
    system_prompt = OVERVIEW_SYSTEM_PROMPT_FULL.format(
        deep_dive_summaries=deep_dive_summaries,
    )

    # Inject editorial stance if provided
    if editorial_stance:
        system_prompt += EDITORIAL_STANCE_SECTION.format(editorial_stance=editorial_stance)

    user_prompt = OVERVIEW_USER_TEMPLATE.format(
        literature_review=lit_review,
        deep_dive_list=deep_dive_list,
    )

    logger.info(
        f"Writing overview referencing {len(deep_dive_drafts)} deep-dives"
    )

    try:
        llm = get_llm(tier=ModelTier.OPUS, max_tokens=MAX_TOKENS)

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

        # Generate a title based on the overview scope
        # For now, use a generic title - could be enhanced with LLM call
        title = _generate_overview_title(overview_scope, lit_review)

        logger.info(
            f"Generated overview: {word_count} words, {len(citation_keys)} citations"
        )

        overview = OverviewDraft(
            title=title,
            content=content,
            word_count=word_count,
            citation_keys=citation_keys,
        )

        return {"overview_draft": overview}

    except Exception as e:
        logger.error(f"Failed to write overview: {e}")
        return {
            "errors": [{"node": "write_overview", "error": str(e)}]
        }


def _generate_overview_title(scope: str, lit_review: str) -> str:
    """Generate a title for the overview based on scope or content.

    For now, uses a simple heuristic. Could be enhanced with LLM call.
    """
    # Try to extract a title-like phrase from the scope
    if scope:
        # Take first sentence and clean it up
        first_sentence = scope.split(".")[0].strip()
        if len(first_sentence) < 60:
            return first_sentence

    # Try to find a header in the literature review
    header_match = re.search(r"^#\s+(.+)$", lit_review, re.MULTILINE)
    if header_match:
        return header_match.group(1).strip()

    return "Overview"
