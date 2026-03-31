"""Overview writing node for evening_reads workflow.

Writes the overview article that synthesizes the big picture and references
the deep-dives without duplicating their content.
"""

import logging
import re
from typing import Any

from langsmith import traceable

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier

from ..prompts import EDITORIAL_STANCE_SECTION, OVERVIEW_SYSTEM_PROMPT_FULL, OVERVIEW_USER_TEMPLATE
from ..state import DeepDiveDraft, OverviewDraft, EveningReadsState, CitationKeyMapping

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
            f'**{draft["id"].replace("_", " ").title()}**: "{draft["title"]}"\nTheme preview: {first_para}...'
        )

    return "\n\n".join(summaries)


def _format_deep_dive_list(drafts: list[DeepDiveDraft]) -> str:
    """Format deep-dive list for the user prompt."""
    if not drafts:
        return "No deep-dives planned."

    items = []
    for i, draft in enumerate(drafts, 1):
        items.append(f'{i}. "{draft["title"]}" ({draft["id"]})')

    return "\n".join(items)


@traceable(run_type="chain", name="EveningReads_WriteOverview")
async def write_overview_node(state: EveningReadsState) -> dict[str, Any]:
    """Write the overview article.

    This node runs after all deep-dives are complete and references them.

    Returns:
        State update with overview_draft
    """
    lit_review = state["input"]["literature_review"]
    editorial_stance = state["input"].get("editorial_stance", "")
    editorial_emphasis = state["input"].get("editorial_emphasis", {})
    wants_recency = editorial_emphasis.get("recency") == "high"
    deep_dive_drafts = state.get("deep_dive_drafts", [])
    right_now_hooks = state.get("right_now_hooks", [])
    overview_scope = state.get("overview_scope", "")
    citation_mappings: dict[str, CitationKeyMapping] = state.get("citation_mappings", {})

    if not deep_dive_drafts:
        logger.warning("No deep-dive drafts available for overview")
        return {"errors": [{"node": "write_overview", "error": "No deep-dive drafts available"}]}

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

    # Build recency annotation for the overview writer (only for recency-focused publications)
    recency_note = ""
    if wants_recency:
        all_keys = _extract_citations(lit_review)
        recent_keys = []
        for key in all_keys:
            mapping = citation_mappings.get(key, {})
            year = mapping.get("year")
            if year and year >= 2025:
                title_str = (mapping.get("title") or "")[:60]
                recent_keys.append(f"[@{key}] ({year}) {title_str}")

        if recent_keys:
            recency_note = (
                "\n\n## Recent Sources Available (2025-2026) — prioritize these\n"
                + "\n".join(f"- {r}" for r in recent_keys)
            )

    # Build right-now hooks section for the overview (aggregated from all deep-dives)
    hooks_section = ""
    if right_now_hooks:
        hook_lines = []
        for hook in right_now_hooks:
            zk = hook.get("zotero_key")
            cite = f" [@{zk}]" if zk else ""
            hook_lines.append(
                f"- **{hook['source_title']}** ({hook['source_date']}){cite}: {hook['finding']}"
            )
        hooks_section = (
            "\n\n## Right Now — Recent Developments (last 2-3 weeks)\n"
            "These recent findings were discovered via web search. Cite them "
            "using [@KEY] format where available to anchor the overview in the current moment.\n\n"
            + "\n".join(hook_lines)
        )

    user_prompt = OVERVIEW_USER_TEMPLATE.format(
        literature_review=lit_review,
        deep_dive_list=deep_dive_list,
    ) + hooks_section + recency_note

    logger.info(
        f"Writing overview referencing {len(deep_dive_drafts)} deep-dives, "
        f"{len(right_now_hooks)} right-now hooks"
    )

    try:
        response = await invoke(
            tier=ModelTier.OPUS,
            system=system_prompt,
            user=user_prompt,
            config=InvokeConfig(
                max_tokens=MAX_TOKENS,
                batch_policy=BatchPolicy.PREFER_BALANCE,
            ),
        )

        content = response.content if isinstance(response.content, str) else str(response.content)
        word_count = len(content.split())
        citation_keys = _extract_citations(content)

        # Generate a title based on the overview scope
        # For now, use a generic title - could be enhanced with LLM call
        title = _generate_overview_title(overview_scope, lit_review)

        logger.info(f"Generated overview: {word_count} words, {len(citation_keys)} citations")

        overview = OverviewDraft(
            title=title,
            content=content,
            word_count=word_count,
            citation_keys=citation_keys,
        )

        return {"overview_draft": overview}

    except Exception as e:
        logger.error(f"Failed to write overview: {e}")
        return {"errors": [{"node": "write_overview", "error": str(e)}]}


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
