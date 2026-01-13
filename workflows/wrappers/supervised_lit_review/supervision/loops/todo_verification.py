"""TODO verification step to filter low-value items before human review.

This module provides a verification step that uses Sonnet to filter out
low-value TODOs before they reach human reviewers. It processes TODOs in
batches and applies aggressive filtering criteria.
"""

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TodoDecision(BaseModel):
    """Decision for a single TODO item."""

    todo: str = Field(description="The original TODO text")
    decision: str = Field(description="KEEP or DISCARD")
    reasoning: str = Field(description="Brief explanation for the decision")


class TodoVerificationResult(BaseModel):
    """Result of TODO verification."""

    keep: list[str] = Field(
        default_factory=list, description="TODOs that should be kept for human review"
    )
    discard: list[str] = Field(
        default_factory=list, description="TODOs filtered out as low-value"
    )
    reasoning: dict[str, str] = Field(
        default_factory=dict, description="Reasoning for each TODO decision"
    )


TODO_VERIFICATION_SYSTEM = """You are reviewing TODOs generated during academic editing to filter out low-value items.

Your job is to be AGGRESSIVE about discarding. Most TODOs in typical academic editing are borderline - those should be DISCARDED.

## Evaluation Criteria

KEEP a TODO only if it would make a STRONG contribution:
- Addresses a genuinely missing critical element
- Fixes an actual error or significant gap
- Would substantially improve reader understanding
- The paper would be notably weaker without addressing it

DISCARD a TODO if ANY of these apply:
- It's about common knowledge that doesn't need citation
- It's a "nice to have" improvement rather than necessary
- It duplicates another TODO or addresses a similar concern
- It's about stylistic preference rather than substance
- It concerns a different section than the one being reviewed
- It's vague or non-actionable ("needs more detail")
- It requests citations for disciplinary common knowledge
- It flags general trend statements or consensus claims
- It's about methodological descriptions that are standard practice

## Important Guidelines

1. When in doubt, DISCARD. Only truly important items should reach human review.
2. A TODO requesting "more evidence" when evidence exists should be DISCARDED.
3. TODOs about well-established facts (dates, names, basic processes) should be DISCARDED.
4. TODOs that would only marginally improve the paper should be DISCARDED.

## Output Format

For each TODO, respond with a JSON object containing:
- todo: The original TODO text
- decision: "KEEP" or "DISCARD"
- reasoning: One sentence explaining why"""

TODO_VERIFICATION_USER = """## Document Context (first 10,000 characters)
{document_excerpt}

## Research Topic
{topic}

## TODOs to Evaluate ({count} items)
{todos_formatted}

Evaluate each TODO and provide your decision. Remember: be aggressive about discarding."""


async def verify_todos(
    todos: list[str],
    document: str,
    topic: str,
    batch_size: int = 30,
    model_name: str = "claude-sonnet-4-20250514",
) -> TodoVerificationResult:
    """Verify TODOs in batches using Sonnet.

    Args:
        todos: List of TODO strings to verify
        document: The full document being reviewed
        topic: The research topic
        batch_size: Number of TODOs to process per batch
        model_name: The model to use for verification

    Returns:
        TodoVerificationResult with keep/discard lists and reasoning
    """
    if not todos:
        return TodoVerificationResult()

    model = ChatAnthropic(model=model_name, temperature=0)

    keep: list[str] = []
    discard: list[str] = []
    reasoning: dict[str, str] = {}

    # Use first 10K chars of document for context
    document_excerpt = document[:10000] if len(document) > 10000 else document

    # Process in batches
    for i in range(0, len(todos), batch_size):
        batch = todos[i : i + batch_size]
        logger.info(
            f"Verifying TODO batch {i // batch_size + 1} "
            f"({len(batch)} items, {i + 1}-{min(i + batch_size, len(todos))} of {len(todos)})"
        )

        # Format TODOs for the prompt
        todos_formatted = "\n".join(
            f"{j + 1}. {todo}" for j, todo in enumerate(batch)
        )

        try:
            # Call the model
            response = await model.ainvoke(
                [
                    {"role": "system", "content": TODO_VERIFICATION_SYSTEM},
                    {
                        "role": "user",
                        "content": TODO_VERIFICATION_USER.format(
                            document_excerpt=document_excerpt,
                            topic=topic,
                            count=len(batch),
                            todos_formatted=todos_formatted,
                        ),
                    },
                ]
            )

            # Parse the response
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", "") if content else ""

            # Parse decisions from response
            batch_decisions = _parse_verification_response(content, batch)

            for todo, decision, reason in batch_decisions:
                reasoning[todo] = reason
                if decision == "KEEP":
                    keep.append(todo)
                else:
                    discard.append(todo)

        except Exception as e:
            logger.error(f"Error verifying TODO batch: {e}")
            # On error, keep all TODOs in this batch (fail safe)
            for todo in batch:
                keep.append(todo)
                reasoning[todo] = f"Kept due to verification error: {e}"

    logger.info(
        f"TODO verification complete: {len(keep)} kept, {len(discard)} discarded"
    )
    return TodoVerificationResult(keep=keep, discard=discard, reasoning=reasoning)


def _parse_verification_response(
    response: str, original_todos: list[str]
) -> list[tuple[str, str, str]]:
    """Parse the model's verification response.

    Returns list of (todo, decision, reasoning) tuples.
    """
    import json
    import re

    results: list[tuple[str, str, str]] = []
    processed_indices: set[int] = set()

    # Try to find JSON objects in the response
    json_pattern = r'\{[^{}]*"decision"[^{}]*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)

    for match in matches:
        try:
            obj = json.loads(match)
            decision = obj.get("decision", "").upper()
            reason = obj.get("reasoning", "No reasoning provided")

            # Try to match to original TODO
            todo_text = obj.get("todo", "")
            matched_idx = _match_todo_to_original(todo_text, original_todos)

            if matched_idx is not None and matched_idx not in processed_indices:
                processed_indices.add(matched_idx)
                results.append(
                    (
                        original_todos[matched_idx],
                        decision if decision in ["KEEP", "DISCARD"] else "KEEP",
                        reason,
                    )
                )
        except json.JSONDecodeError:
            continue

    # For any TODOs not matched, default to KEEP
    for i, todo in enumerate(original_todos):
        if i not in processed_indices:
            results.append((todo, "KEEP", "Could not parse verification response"))

    return results


def _match_todo_to_original(
    todo_text: str, original_todos: list[str]
) -> int | None:
    """Match a TODO text from response to the original list.

    Uses fuzzy matching to handle minor variations.
    """
    if not todo_text:
        return None

    # Normalize for comparison
    normalized_text = todo_text.lower().strip()

    # Try exact match first
    for i, orig in enumerate(original_todos):
        if orig.lower().strip() == normalized_text:
            return i

    # Try substring match
    for i, orig in enumerate(original_todos):
        orig_lower = orig.lower().strip()
        if normalized_text in orig_lower or orig_lower in normalized_text:
            return i

    # Try matching by significant words
    text_words = set(normalized_text.split())
    best_match = None
    best_overlap = 0

    for i, orig in enumerate(original_todos):
        orig_words = set(orig.lower().split())
        overlap = len(text_words & orig_words)
        if overlap > best_overlap and overlap >= 3:
            best_overlap = overlap
            best_match = i

    return best_match


def verify_todos_sync(
    todos: list[str],
    document: str,
    topic: str,
    batch_size: int = 30,
) -> TodoVerificationResult:
    """Synchronous wrapper for verify_todos."""
    import asyncio

    return asyncio.run(verify_todos(todos, document, topic, batch_size))
