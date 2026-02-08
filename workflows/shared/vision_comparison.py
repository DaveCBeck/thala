"""Shared vision-based pair comparison for image candidate selection.

Uses MLLM pair comparison (80.6% accuracy per MLLM-as-a-Judge research,
vs 55.7% for scoring-based evaluation) to select the best image from
a set of candidates.

Call sites:
- image_utils.py: Imagen multi-candidate selection (A3)
- diagram_utils/mermaid.py: Mermaid candidate selection (B5, future)
"""

import base64
import logging

from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)


PAIR_COMPARISON_SYSTEM = """You are comparing two images to select the better one.

Evaluate based on the selection criteria provided. Consider:
1. How well each image matches the stated requirements
2. Visual quality (composition, clarity, professional appearance)
3. Overall effectiveness for the intended purpose

Respond with ONLY "A" or "B" — nothing else."""


PAIR_COMPARISON_USER = """Which image better matches these criteria?

**Selection Criteria:**
{criteria}

Image A and Image B are attached. Respond with ONLY "A" or "B"."""


async def vision_pair_select(
    candidates: list[bytes],
    selection_criteria: str,
    model_tier: ModelTier = ModelTier.SONNET,
) -> int:
    """Select the best candidate image via vision-based pair comparison.

    Uses tournament-style pair comparison to find the best image from
    a list of candidates. More accurate than scoring-based evaluation.

    Args:
        candidates: List of PNG image bytes to compare
        selection_criteria: What to evaluate against (brief, prompt, etc.)
        model_tier: Vision model tier to use

    Returns:
        0-based index of the best candidate

    Falls back to index 0 if vision comparison fails.
    """
    if len(candidates) <= 1:
        return 0

    try:
        # Tournament: compare sequentially, winner advances
        current_best_idx = 0

        for challenger_idx in range(1, len(candidates)):
            winner = await _compare_pair(
                candidates[current_best_idx],
                candidates[challenger_idx],
                selection_criteria,
                model_tier,
            )
            if winner == "B":
                current_best_idx = challenger_idx

        logger.info(f"Vision pair selection chose candidate {current_best_idx + 1} of {len(candidates)}")
        return current_best_idx

    except Exception as e:
        logger.warning(f"Vision pair selection failed, using first candidate: {e}")
        return 0


async def _compare_pair(
    image_a: bytes,
    image_b: bytes,
    criteria: str,
    model_tier: ModelTier,
) -> str:
    """Compare two images, return 'A' or 'B'."""
    llm = get_llm(tier=model_tier, max_tokens=16)

    b64_a = base64.b64encode(image_a).decode("utf-8")
    b64_b = base64.b64encode(image_b).decode("utf-8")

    content_parts = [
        {"type": "text", "text": PAIR_COMPARISON_USER.format(criteria=criteria)},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64_a,
            },
        },
        {"type": "text", "text": "Image A is shown above. Image B is shown below."},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64_b,
            },
        },
    ]

    response = await llm.ainvoke(
        [
            {"role": "system", "content": PAIR_COMPARISON_SYSTEM},
            {"role": "user", "content": content_parts},
        ]
    )

    answer = (response.content if isinstance(response.content, str) else str(response.content)).strip().upper()

    # Parse response — accept "A" or "B" (possibly with extra text)
    if answer.startswith("A"):
        return "A"
    if answer.startswith("B"):
        return "B"

    # Default to A if response is ambiguous
    logger.warning(f"Ambiguous pair comparison response: {answer!r}, defaulting to A")
    return "A"
