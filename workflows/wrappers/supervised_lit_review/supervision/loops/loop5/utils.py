"""Utility functions for Loop 5 fact and reference checking."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

from workflows.shared.token_utils import (
    estimate_tokens_fast,
    estimate_request_tokens,
    CHARS_PER_TOKEN,
    DEFAULT_RESPONSE_BUFFER,
    HAIKU_SAFE_LIMIT,
    SONNET_1M_SAFE_LIMIT,
)

HAIKU_MAX_TOKENS = HAIKU_SAFE_LIMIT
SONNET_1M_MAX_TOKENS = SONNET_1M_SAFE_LIMIT
SONNET_1M_THRESHOLD = 120_000
RESPONSE_BUFFER_TOKENS = DEFAULT_RESPONSE_BUFFER


def estimate_loop5_request_tokens(
    section_content: str,
    system_prompt: str,
    paper_summaries_text: str,
    include_tools: bool = True,
) -> int:
    """Estimate total tokens for a Loop 5 LLM request."""
    combined_user_content = f"{section_content}\n\n{paper_summaries_text}"

    return estimate_request_tokens(
        user_prompt=combined_user_content,
        system_prompt=system_prompt,
        message_count=2,
        include_tool_definitions=include_tools,
        response_buffer=RESPONSE_BUFFER_TOKENS,
    )


def calculate_dynamic_char_budget(
    section_content: str,
    system_prompt: str,
    num_sections: int,
    target_max_tokens: int = HAIKU_MAX_TOKENS,
) -> int:
    """Calculate dynamic character budget for paper context per section."""
    section_tokens = estimate_tokens_fast(section_content, with_safety_margin=False)
    system_tokens = estimate_tokens_fast(system_prompt, with_safety_margin=False)
    base_tokens = section_tokens + system_tokens + RESPONSE_BUFFER_TOKENS

    available_tokens = target_max_tokens - base_tokens
    per_section_tokens = int(available_tokens * 0.8 / max(num_sections, 1))
    available_chars = per_section_tokens * CHARS_PER_TOKEN

    return max(5000, min(available_chars, 20000))


def format_paper_summaries(paper_summaries: dict[str, Any]) -> str:
    """Format paper summaries for prompt context."""
    if not paper_summaries:
        return "No paper summaries available."

    lines = []
    for doi, summary in paper_summaries.items():
        lines.append(f"DOI: {doi}")
        lines.append(f"Title: {summary.get('title', 'N/A')}")
        lines.append(f"Authors: {', '.join(summary.get('authors', []))}")
        lines.append(f"Year: {summary.get('year', 'N/A')}")
        lines.append(f"Summary: {summary.get('short_summary', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def format_citation_keys(zotero_keys: dict[str, str]) -> str:
    """Format citation keys for reference checking."""
    if not zotero_keys:
        return "No citation keys available."

    lines = []
    for doi, key in zotero_keys.items():
        lines.append(f"[@{key}] -> {doi}")

    return "\n".join(lines)
