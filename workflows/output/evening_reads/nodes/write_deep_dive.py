"""Deep-dive writing node for evening_reads workflow.

Writes individual deep-dive articles with distinctiveness enforcement.
Each deep-dive gets a "must avoid" list of themes covered by other deep-dives.
"""

import logging
import re
from typing import Any, Literal

from langsmith import traceable

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier

from ..prompts import (
    DEEP_DIVE_PUZZLE_PROMPT_FULL,
    DEEP_DIVE_FINDING_PROMPT_FULL,
    DEEP_DIVE_CONTRARIAN_PROMPT_FULL,
    DEEP_DIVE_MECHANISM_PROMPT_FULL,
    DEEP_DIVE_NARRATIVE_PROMPT_FULL,
    DEEP_DIVE_COMPARISON_PROMPT_FULL,
    DEEP_DIVE_OPEN_PROMPT_FULL,
    DEEP_DIVE_USER_TEMPLATE,
    EDITORIAL_STANCE_SECTION,
)
from ..state import DeepDiveDraft, EnrichedContent

# Map structural approach to prompt
STRUCTURAL_PROMPTS = {
    "puzzle": DEEP_DIVE_PUZZLE_PROMPT_FULL,
    "finding": DEEP_DIVE_FINDING_PROMPT_FULL,
    "contrarian": DEEP_DIVE_CONTRARIAN_PROMPT_FULL,
    "mechanism": DEEP_DIVE_MECHANISM_PROMPT_FULL,
    "narrative": DEEP_DIVE_NARRATIVE_PROMPT_FULL,
    "comparison": DEEP_DIVE_COMPARISON_PROMPT_FULL,
    "open": DEEP_DIVE_OPEN_PROMPT_FULL,
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


def _extract_current_year_contexts(
    lit_review: str, citation_mappings: dict, already_in_excerpt: str
) -> str:
    """Extract paragraphs containing current-year citations not already in the excerpt.

    This ensures 2026 sources are visible to the writer regardless of section boundaries.
    """
    import datetime

    current_year = datetime.date.today().year
    current_year_keys = {
        k for k, m in citation_mappings.items()
        if (m.get("year") or 0) >= current_year
    }

    if not current_year_keys:
        return ""

    # Find which current-year keys are NOT already mentioned in the excerpt
    missing_keys = {k for k in current_year_keys if f"[@{k}]" not in already_in_excerpt}
    if not missing_keys:
        return ""

    # Extract paragraphs containing these keys
    paragraphs = lit_review.split("\n\n")
    relevant_paras = []
    for para in paragraphs:
        for key in missing_keys:
            if f"[@{key}]" in para or f"@{key}" in para:
                relevant_paras.append(para.strip())
                break

    if not relevant_paras:
        return ""

    return (
        "\n\n### Additional Recent Findings (2026)\n\n"
        + "\n\n".join(relevant_paras)
    )


def _extract_relevant_sections(lit_review: str, section_names: list[str]) -> str:
    """Extract sections from the literature review that match the given names.

    Falls back to returning the full review if no sections match.
    """
    if not section_names:
        return lit_review

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
            matches_section = any(section.lower() in header_text.lower() for section in section_names)

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
        logger.info(f"Extracted {len(excerpt)} chars from {len(matched_headers)} sections: {matched_headers}")
        return excerpt

    # Fallback: return full review
    logger.info(f"No matching sections found; returning full review ({len(lit_review)} chars)")
    return lit_review


@traceable(run_type="chain", name="EveningReads_WriteDeepDive")
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
    deep_dive_id: Literal["deep_dive_1", "deep_dive_2", "deep_dive_3"] = state.get("deep_dive_id")
    title = state.get("title", "Untitled")
    theme = state.get("theme", "")
    structural_approach = state.get("structural_approach", "puzzle")
    relevant_sections = state.get("relevant_sections", [])
    must_avoid = state.get("must_avoid", [])
    enriched_content: list[EnrichedContent] = state.get("enriched_content", [])
    right_now_hooks = state.get("right_now_hooks", [])
    lit_review = state.get("literature_review", "")
    editorial_stance = state.get("editorial_stance", "")
    editorial_emphasis = state.get("editorial_emphasis", {})
    wants_recency = editorial_emphasis.get("recency") == "high"
    citation_mappings = state.get("citation_mappings", {})

    if not deep_dive_id:
        return {"errors": [{"node": "write_deep_dive", "error": "Missing deep_dive_id"}]}

    # Build source content from enriched content for this deep-dive
    source_parts = []
    for ec in enriched_content:
        if ec["deep_dive_id"] == deep_dive_id:
            mapping = citation_mappings.get(ec["zotero_key"], {})
            year = mapping.get("year")
            year_str = f", {year}" if year else ""
            source_parts.append(
                f"## Source: {ec['zotero_key']} ({ec['content_level']}{year_str})\n\n{ec['content']}"
            )

    source_content = "\n\n---\n\n".join(source_parts) if source_parts else "[No source content available]"

    # Extract relevant sections from the literature review
    lit_review_excerpt = _extract_relevant_sections(lit_review, relevant_sections)

    # Append paragraphs with current-year citations not already in the excerpt
    if wants_recency:
        cy_contexts = _extract_current_year_contexts(lit_review, citation_mappings, lit_review_excerpt)
        if cy_contexts:
            lit_review_excerpt += cy_contexts
            logger.info(f"Appended {len(cy_contexts)} chars of current-year citation context to excerpt")

    # Build must_avoid string
    must_avoid_str = "\n".join(f"- {item}" for item in must_avoid) if must_avoid else "None specified"

    # Select prompt based on structural approach
    prompt_template = STRUCTURAL_PROMPTS.get(structural_approach, STRUCTURAL_PROMPTS["puzzle"])

    # Format system prompt with assignment details
    system_prompt = prompt_template.format(
        title=title,
        theme=theme,
        must_avoid=must_avoid_str,
    )

    # Inject editorial stance if provided
    if editorial_stance:
        system_prompt += EDITORIAL_STANCE_SECTION.format(editorial_stance=editorial_stance)

    # Build recency annotation: identify which citations in the excerpt are recent
    recency_note = ""
    if wants_recency:
        excerpt_keys = _extract_citations(lit_review_excerpt)
        recent_in_excerpt = []
        older_in_excerpt = []
        for key in excerpt_keys:
            mapping = citation_mappings.get(key, {})
            year = mapping.get("year")
            title_str = mapping.get("title", "")
            label = f"[@{key}]"
            if year:
                label += f" ({year})"
            if title_str:
                label += f" {title_str[:60]}"
            if year and year >= 2025:
                recent_in_excerpt.append(label)
            elif year:
                older_in_excerpt.append(label)

        if recent_in_excerpt:
            recency_note = (
                "\n\n## Recent Sources in This Excerpt (prioritize these)\n"
                + "\n".join(f"- {r}" for r in recent_in_excerpt)
            )
            if older_in_excerpt:
                recency_note += (
                    "\n\n## Older Sources (use for context, not as primary evidence)\n"
                    + "\n".join(f"- {o}" for o in older_in_excerpt)
                )

    # Build right-now hooks section if available
    hooks_section = ""
    if right_now_hooks:
        hook_parts = []
        for hook in right_now_hooks:
            zk = hook.get("zotero_key")
            cite_hint = f"Cite as [@{zk}]" if zk else "Name the source and date inline"
            entry = (
                f"### {hook['source_title']} ({hook['source_date']})\n"
                f"{cite_hint}\n\n"
                f"**Why this matters for your piece:** {hook['finding']}\n\n"
            )
            if hook.get("content"):
                entry += f"**Source content:**\n\n{hook['content']}\n"
            hook_parts.append(entry)

        hooks_section = (
            "\n\n## Right Now — Recent Developments (last 2-3 weeks)\n"
            "These are concrete recent findings discovered via web search. "
            "Use them to anchor your opening — lead with what just happened, "
            "then connect to the deeper literature review material.\n\n"
            "Cite these sources using the [@KEY] format shown above where available. "
            "They will appear in the References section alongside the academic sources.\n\n"
            + "\n\n---\n\n".join(hook_parts)
        )

    user_prompt = DEEP_DIVE_USER_TEMPLATE.format(
        source_content=source_content,
        literature_review_excerpt=lit_review_excerpt,
    ) + hooks_section + recency_note

    logger.info(
        f"Writing deep-dive {deep_dive_id}: '{title}' "
        f"(approach={structural_approach}, {len(source_parts)} sources, "
        f"{len(must_avoid)} themes to avoid, {len(right_now_hooks)} right-now hooks)"
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

        logger.info(f"Generated deep-dive {deep_dive_id}: {word_count} words, {len(citation_keys)} citations")

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
        return {"errors": [{"node": f"write_deep_dive_{deep_dive_id}", "error": str(e)}]}
