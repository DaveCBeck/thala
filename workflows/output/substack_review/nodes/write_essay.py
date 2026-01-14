"""Essay writing nodes - three parallel agents with distinct angles."""

import logging
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic

from workflows.shared.llm_utils import ModelTier
from ..state import EssayDraft
from ..prompts import (
    PUZZLE_SYSTEM_PROMPT,
    PUZZLE_USER_TEMPLATE,
    FINDING_SYSTEM_PROMPT,
    FINDING_USER_TEMPLATE,
    CONTRARIAN_SYSTEM_PROMPT,
    CONTRARIAN_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Target 3000-4000 words = ~12000-16000 tokens output
MAX_TOKENS = 16000


async def _write_essay(
    literature_review: str,
    angle: Literal["puzzle", "finding", "contrarian"],
    system_prompt: str,
    user_template: str,
) -> EssayDraft:
    """Write a single essay with the given angle.

    Uses OPUS for high-quality long-form output.
    """
    llm = ChatAnthropic(
        model=ModelTier.OPUS.value,
        max_tokens=MAX_TOKENS,
    )

    user_prompt = user_template.format(literature_review=literature_review)

    logger.info(f"Generating {angle} essay with OPUS")

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    content = (
        response.content if isinstance(response.content, str) else str(response.content)
    )
    word_count = len(content.split())

    logger.info(f"Generated {angle} essay: {word_count} words")

    return EssayDraft(
        angle=angle,
        content=content,
        word_count=word_count,
    )


async def write_puzzle_essay(state: dict) -> dict[str, Any]:
    """Write essay using Narrative Entry Through a Specific Puzzle angle.

    This node receives state via Send() with literature_review key.
    """
    lit_review = state.get("literature_review") or state["input"]["literature_review"]

    try:
        essay = await _write_essay(
            literature_review=lit_review,
            angle="puzzle",
            system_prompt=PUZZLE_SYSTEM_PROMPT,
            user_template=PUZZLE_USER_TEMPLATE,
        )
        return {"essay_drafts": [essay]}
    except Exception as e:
        logger.error(f"Failed to write puzzle essay: {e}")
        return {
            "errors": [{"node": "write_puzzle_essay", "error": str(e)}],
        }


async def write_finding_essay(state: dict) -> dict[str, Any]:
    """Write essay using Lead With the Striking Empirical Finding angle.

    This node receives state via Send() with literature_review key.
    """
    lit_review = state.get("literature_review") or state["input"]["literature_review"]

    try:
        essay = await _write_essay(
            literature_review=lit_review,
            angle="finding",
            system_prompt=FINDING_SYSTEM_PROMPT,
            user_template=FINDING_USER_TEMPLATE,
        )
        return {"essay_drafts": [essay]}
    except Exception as e:
        logger.error(f"Failed to write finding essay: {e}")
        return {
            "errors": [{"node": "write_finding_essay", "error": str(e)}],
        }


async def write_contrarian_essay(state: dict) -> dict[str, Any]:
    """Write essay using Contrarian Framing angle.

    This node receives state via Send() with literature_review key.
    """
    lit_review = state.get("literature_review") or state["input"]["literature_review"]

    try:
        essay = await _write_essay(
            literature_review=lit_review,
            angle="contrarian",
            system_prompt=CONTRARIAN_SYSTEM_PROMPT,
            user_template=CONTRARIAN_USER_TEMPLATE,
        )
        return {"essay_drafts": [essay]}
    except Exception as e:
        logger.error(f"Failed to write contrarian essay: {e}")
        return {
            "errors": [{"node": "write_contrarian_essay", "error": str(e)}],
        }
