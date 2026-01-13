"""Retry logic and LLM fallback for structural editing."""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_structured_output, get_llm
from workflows.supervised_lit_review.supervision.types import EditManifest
from workflows.supervised_lit_review.supervision.prompts import (
    LOOP3_RETRY_SYSTEM,
    LOOP3_EDITOR_SYSTEM,
    LOOP3_EDITOR_USER,
)

logger = logging.getLogger(__name__)


async def retry_analyze_node(state: dict) -> dict[str, Any]:
    """Retry analysis specifically for edits missing replacement_text.

    Re-prompts the LLM with explicit instructions to provide replacement_text
    for trim_redundancy and split_section edits.
    """
    needs_retry_edits = state.get("needs_retry_edits", [])
    numbered_doc = state["numbered_document"]
    input_data = state["input"]
    topic = input_data.get("topic", "")

    retry_prompt = f"""The previous analysis identified edits that require replacement_text but it was missing.

Please provide the replacement_text for these specific edits:

"""
    for edit in needs_retry_edits:
        edit_type = edit.get("edit_type")
        source = edit.get("source_paragraph")
        notes = edit.get("notes", "")
        retry_prompt += f"""
- {edit_type} for P{source}: {notes}
  Please provide the replacement_text (the trimmed/split content).
"""

    retry_prompt += f"""
## Numbered Document
{numbered_doc}

## Research Topic
{topic}

Provide an EditManifest with the SAME edits but include replacement_text for each."""

    try:
        manifest = await get_structured_output(
            output_schema=EditManifest,
            user_prompt=retry_prompt,
            system_prompt=LOOP3_RETRY_SYSTEM,
            tier=ModelTier.OPUS,
            thinking_budget=4000,
            max_tokens=8096,
            use_json_schema_method=True,
            max_retries=1,
        )

        logger.debug(f"Retry analysis: {len(manifest.edits)} edits returned")

        return {
            "edit_manifest": manifest.model_dump(),
            "retry_attempted": True,
        }

    except Exception as e:
        logger.error(f"Retry analysis failed: {e}", exc_info=True)
        return {"retry_attempted": True}


def _format_manifest(manifest: dict) -> str:
    """Format edit manifest for the editor prompt."""
    lines = []

    lines.append(f"Overall Assessment: {manifest.get('overall_assessment', 'N/A')}")
    lines.append("")

    edits = manifest.get("edits", [])
    if edits:
        lines.append("Structural Edits:")
        for i, edit in enumerate(edits, 1):
            edit_type = edit.get("edit_type", "unknown")
            source = edit.get("source_paragraph", "?")
            target = edit.get("target_paragraph")
            notes = edit.get("notes", "")

            lines.append(f"{i}. {edit_type.upper()}")
            lines.append(f"   Source: [P{source}]")
            if target is not None:
                lines.append(f"   Target: [P{target}]")
            lines.append(f"   Notes: {notes}")
            lines.append("")

    todo_markers = manifest.get("todo_markers", [])
    if todo_markers:
        lines.append("TODO Markers to Insert:")
        for i, todo in enumerate(todo_markers, 1):
            lines.append(f"{i}. <!-- TODO: {todo} -->")
        lines.append("")

    return "\n".join(lines)


async def execute_manifest_node(state: dict) -> dict[str, Any]:
    """Execute the edit manifest to restructure the document (LLM fallback).

    Used when programmatic edit application fails or isn't possible.
    Uses Opus to carefully execute structural changes while preserving
    citations and academic formatting.
    """
    numbered_doc = state["numbered_document"]
    manifest = state.get("edit_manifest")

    if not manifest:
        logger.warning("No manifest to execute")
        return {"fallback_used": True}

    manifest_text = _format_manifest(manifest)

    user_prompt = LOOP3_EDITOR_USER.format(
        numbered_document=numbered_doc,
        edit_manifest=manifest_text,
    )

    llm = get_llm(
        tier=ModelTier.OPUS,
        max_tokens=16384,
    )

    try:
        messages = [
            {"role": "system", "content": LOOP3_EDITOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        response = await llm.ainvoke(messages)

        restructured_text = ""
        if isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    restructured_text = block.get("text", "")
                    break
                elif hasattr(block, "type") and block.type == "text":
                    restructured_text = getattr(block, "text", "")
                    break
        else:
            restructured_text = response.content

        logger.info("Successfully executed edit manifest (LLM fallback)")

        return {
            "current_review": restructured_text,
            "fallback_used": True,
        }

    except Exception as e:
        logger.error(f"Manifest execution failed: {e}", exc_info=True)
        return {"fallback_used": True}
