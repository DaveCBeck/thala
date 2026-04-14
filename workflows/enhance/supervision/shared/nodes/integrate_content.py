"""Integrate content node for supervision loop."""

import logging
from typing import Any

from langsmith import traceable

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier
from workflows.shared.llm_utils.integration_guard import call_text_with_guards
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

    Uses Opus to integrate new theoretical content into the existing review,
    following the supervisor's integration guidance.

    On unrecoverable shrinkage or any integrator error, the exception
    propagates so the task fails (the incremental checkpoint is retained
    for manual recovery). No soft-fail path.
    """
    current_review = state.get("current_review", "")
    decision = state.get("decision", {})
    expansion_result = state.get("expansion_result", {})
    iteration = state.get("iteration", 0)

    issue = decision.get("issue", {})
    if not issue:
        logger.warning("No issue details for integration, skipping")
        return {"iteration": iteration + 1}

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

    summaries_text = _format_paper_summaries(paper_summaries, zotero_keys)
    citation_keys_text = _format_citation_keys(zotero_keys)

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

    # call_text_with_guards handles both (a) stop_reason=max_tokens
    # continuations and (b) self-condensation retries. Raises
    # IntegrationShrinkageError if the review can't be preserved — we do
    # NOT catch it: the task should fail cleanly rather than produce a
    # silently-truncated review that downstream loops then build on.
    integrated_review = await call_text_with_guards(
        input_content=current_review,
        tier=ModelTier.OPUS,
        system=INTEGRATOR_SYSTEM,
        user=user_prompt,
        config=InvokeConfig(
            effort="high",
            max_tokens=64000,
            cache=False,
            batch_policy=BatchPolicy.PREFER_BALANCE,
        ),
        label=f"loop1_integrator[{topic[:40]}]",
    )

    duplicates = detect_duplicate_headers(integrated_review)
    if duplicates:
        logger.info(f"Removing {len(duplicates)} duplicate headers after Loop 1 integration")
        integrated_review = remove_duplicate_headers(integrated_review, duplicates)

    logger.info(
        f"Integration complete for '{topic}': {len(processed_dois)} papers integrated "
        f"(input={len(current_review)} chars, output={len(integrated_review)} chars)"
    )

    checkpoint_callback = state.get("checkpoint_callback")
    if checkpoint_callback:
        result = checkpoint_callback(
            iteration + 1,
            {
                "current_review": integrated_review,
                "iteration": iteration + 1,
                "supervision_expansions": state.get("supervision_expansions", [])
                + [
                    {
                        "iteration": iteration,
                        "topic": topic,
                        "issue_type": issue_type,
                        "research_query": issue.get("research_query", ""),
                        "papers_added": processed_dois,
                        "integration_summary": f"Integrated {len(processed_dois)} papers on {topic}",
                    }
                ],
                "paper_corpus": state.get("paper_corpus", {}),
                "paper_summaries": state.get("paper_summaries", {}),
                "zotero_keys": state.get("zotero_keys", {}),
            },
        )
        if hasattr(result, "__await__"):
            await result
        logger.debug(f"Supervision checkpoint saved at iteration {iteration + 1}")

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

        zotero_key = zotero_keys.get(doi) or summary.get("zotero_key", "")

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
