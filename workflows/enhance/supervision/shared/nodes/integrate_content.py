"""Integrate content node for supervision loop."""

import logging
from typing import Any

from langsmith import traceable

from workflows.shared.llm_utils.models import ModelTier, get_llm
from workflows.enhance.supervision.shared.prompts import (
    INTEGRATOR_SYSTEM,
    INTEGRATOR_USER,
)
from workflows.enhance.supervision.shared.utils import (
    detect_duplicate_headers,
    remove_duplicate_headers,
)

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="SupervisionIntegrateContent")
async def integrate_content_node(state: dict[str, Any]) -> dict[str, Any]:
    """Integrate expansion findings into the literature review.

    Uses Opus to intelligently integrate new theoretical content into
    the existing review, following the supervisor's integration guidance.

    Args:
        state: Current supervision state containing:
            - current_review: The current literature review text
            - decision: Supervisor decision with issue details and guidance
            - expansion_result: Results from focused expansion
            - iteration: Current iteration number

    Returns:
        State updates including:
            - current_review: Updated review with integrated content
            - supervision_expansions: Record of this integration
            - iteration: Incremented iteration counter
    """
    current_review = state.get("current_review", "")
    decision = state.get("decision", {})
    expansion_result = state.get("expansion_result", {})
    iteration = state.get("iteration", 0)

    issue = decision.get("issue", {})
    if not issue:
        logger.warning("No issue details for integration, skipping")
        return {
            "iteration": iteration + 1,
        }

    topic = issue.get("topic", "")
    issue_type = issue.get("issue_type", "")
    rationale = issue.get("rationale", "")
    related_section = issue.get("related_section", "")
    integration_guidance = issue.get("integration_guidance", "")

    paper_summaries = expansion_result.get("paper_summaries", {})
    zotero_keys = expansion_result.get("zotero_keys", {})
    processed_dois = expansion_result.get("processed_dois", [])

    if not paper_summaries:
        logger.warning("No paper summaries to integrate")
        return {
            "iteration": iteration + 1,
            "supervision_expansions": [
                {
                    "iteration": iteration,
                    "topic": topic,
                    "issue_type": issue_type,
                    "research_query": issue.get("research_query", ""),
                    "papers_added": [],
                    "integration_summary": "No papers found to integrate",
                }
            ],
        }

    # Format paper summaries for the integrator (with zotero keys inline)
    summaries_text = _format_paper_summaries(paper_summaries, zotero_keys)

    # Format citation keys as quick reference (also inline in summaries above)
    citation_keys_text = _format_citation_keys(zotero_keys)

    # Build the user prompt
    user_prompt = INTEGRATOR_USER.format(
        current_review=current_review,
        topic=topic,
        issue_type=issue_type,
        rationale=rationale,
        related_section=related_section,
        integration_guidance=integration_guidance,
        paper_summaries=summaries_text,
        new_citation_keys=citation_keys_text,
    )

    # Use Opus for complex integration with large output capacity
    llm = get_llm(
        tier=ModelTier.OPUS,
        thinking_budget=8000,
        max_tokens=32000,
    )

    try:
        messages = [
            {"role": "system", "content": INTEGRATOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        response = await llm.ainvoke(messages)

        # Extract the integrated review from response
        integrated_review = _extract_review_content(response)

        # Clean up any duplicate headers introduced during integration
        duplicates = detect_duplicate_headers(integrated_review)
        if duplicates:
            logger.info(
                f"Removing {len(duplicates)} duplicate headers after Loop 1 integration"
            )
            integrated_review = remove_duplicate_headers(integrated_review, duplicates)

        logger.info(
            f"Integration complete for '{topic}': "
            f"{len(processed_dois)} papers integrated"
        )

        return {
            "current_review": integrated_review,
            "iteration": iteration + 1,
            "supervision_expansions": [
                {
                    "iteration": iteration,
                    "topic": topic,
                    "issue_type": issue_type,
                    "research_query": issue.get("research_query", ""),
                    "papers_added": processed_dois,
                    "integration_summary": f"Integrated {len(processed_dois)} papers on {topic}",
                }
            ],
        }

    except Exception as e:
        logger.error(f"Integration failed: {e}")
        # On error, keep current review but DON'T increment iteration
        return {
            "integration_failed": True,
            "supervision_expansions": [
                {
                    "iteration": iteration,
                    "topic": topic,
                    "issue_type": issue_type,
                    "research_query": issue.get("research_query", ""),
                    "papers_added": [],
                    "integration_summary": f"Integration failed: {e}",
                    "failed": True,
                }
            ],
            "loop_error": {
                "loop_number": 1,
                "iteration": iteration,
                "node_name": "integrate_content",
                "error_type": "integration_error",
                "error_message": str(e),
                "recoverable": True,
            },
        }


def _format_paper_summaries(
    paper_summaries: dict[str, dict],
    zotero_keys: dict[str, str],
) -> str:
    """Format paper summaries for the integrator prompt.

    Each paper header includes its [@ZOTERO_KEY] for easy citation.
    """
    if not paper_summaries:
        return "No paper summaries available"

    lines = []
    for doi, summary in paper_summaries.items():
        title = summary.get("title", "Unknown title")
        authors = summary.get("authors", [])
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        year = summary.get("year", "n.d.")
        short_summary = summary.get("short_summary", "")
        key_findings = summary.get("key_findings", [])

        # Get zotero key - from dict or from summary field
        zotero_key = zotero_keys.get(doi) or summary.get("zotero_key", "")

        # Header with citation key prominently displayed
        if zotero_key:
            lines.append(f"### [@{zotero_key}] {title} ({year})")
        else:
            lines.append(f"### {title} ({year})")

        lines.append(f"**Authors:** {author_str}")
        lines.append(f"**DOI:** {doi}")
        if zotero_key:
            lines.append(f"**Cite as:** [@{zotero_key}]")
        if short_summary:
            lines.append(f"**Summary:** {short_summary}")
        if key_findings:
            lines.append("**Key Findings:**")
            for finding in key_findings[:3]:
                lines.append(f"  - {finding}")
        lines.append("")

    return "\n".join(lines)


def _format_citation_keys(zotero_keys: dict[str, str]) -> str:
    """Format citation keys for reference in text."""
    if not zotero_keys:
        return "No citation keys available"

    lines = ["Use these citation keys to cite the new papers:"]
    for doi, key in zotero_keys.items():
        lines.append(f"  - {doi}: [@{key}]")

    return "\n".join(lines)


def _extract_review_content(response: Any) -> str:
    """Extract review text from LLM response."""
    if hasattr(response, "content"):
        content = response.content
        # Handle list of content blocks (e.g., with thinking)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
                elif isinstance(block, str):
                    return block
        elif isinstance(content, str):
            return content

    # Fallback
    return str(response)
