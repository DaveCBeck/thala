"""Expand topic node for supervision loop."""

import logging
from typing import Any

from workflows.enhance.supervision.shared.focused_expansion import (
    run_focused_expansion,
)

logger = logging.getLogger(__name__)


async def expand_topic_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run focused expansion on the identified theoretical gap.

    Takes the issue identified by the supervisor and runs discovery,
    diffusion, and processing phases to gather relevant papers.

    Args:
        state: Current supervision state containing:
            - decision: Supervisor decision with identified issue
            - topic: Parent topic
            - quality_settings: Inherited quality settings

    Returns:
        State updates including:
            - expansion_result: Papers and summaries from focused expansion
            - paper_corpus: Merged with new papers
            - paper_summaries: Merged with new summaries
            - zotero_keys: Merged with new keys
    """
    decision = state.get("decision", {})
    issue = decision.get("issue", {})

    if not issue:
        logger.warning("No issue to expand, skipping expansion")
        return {
            "expansion_result": None,
        }

    issue_topic = issue.get("topic", "")
    research_query = issue.get("research_query", issue_topic)
    issue_type = issue.get("issue_type", "underlying_theory")

    quality_settings = state.get("quality_settings", {})
    parent_topic = state.get("topic", "")

    logger.info(f"Expanding on identified issue: {issue_topic}")
    logger.info(f"Issue type: {issue_type}")

    try:
        expansion_result = await run_focused_expansion(
            topic=issue_topic,
            research_query=research_query,
            quality_settings=quality_settings,
            parent_topic=parent_topic,
        )

        new_corpus = expansion_result.get("paper_corpus", {})
        new_summaries = expansion_result.get("paper_summaries", {})
        new_zotero_keys = expansion_result.get("zotero_keys", {})
        processed_dois = expansion_result.get("processed_dois", [])

        logger.info(
            f"Expansion complete: {len(new_corpus)} new papers, "
            f"{len(new_summaries)} summaries, {len(processed_dois)} processed"
        )

        return {
            "expansion_result": {
                "topic": issue_topic,
                "issue_type": issue_type,
                "paper_summaries": new_summaries,
                "zotero_keys": new_zotero_keys,
                "processed_dois": processed_dois,
            },
            # These will be merged by reducers
            "paper_corpus": new_corpus,
            "paper_summaries": new_summaries,
            "zotero_keys": new_zotero_keys,
        }

    except Exception as e:
        logger.error(f"Focused expansion failed: {e}")
        iteration = state.get("iteration", 0)
        return {
            "expansion_result": {
                "topic": issue_topic,
                "issue_type": issue_type,
                "error": str(e),
                "paper_summaries": {},
                "zotero_keys": {},
                "processed_dois": [],
                "failed": True,
            },
            # Return empty dicts for consistency with success case
            # These will merge with nothing, preserving existing state
            "paper_corpus": {},
            "paper_summaries": {},
            "zotero_keys": {},
            "expansion_failed": True,
            "loop_error": {
                "loop_number": 1,
                "iteration": iteration,
                "node_name": "expand_topic",
                "error_type": "expansion_error",
                "error_message": str(e),
                "recoverable": True,
            },
        }
