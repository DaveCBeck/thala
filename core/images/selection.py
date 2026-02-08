"""Optional LLM-based image selection."""

import logging

from pydantic import BaseModel, Field

from workflows.shared.llm_utils import ModelTier, invoke

from .types import ImageResult

logger = logging.getLogger(__name__)


class ImageSelection(BaseModel):
    """LLM output for image selection."""

    selected_index: int = Field(description="0-based index of best image")
    brief_compliance_score: int = Field(
        ge=1, le=5, description="How well the image matches the brief's explicit requests"
    )
    mood_score: int = Field(ge=1, le=5, description="How well the image's mood matches the brief's tone")
    quality_score: int = Field(ge=1, le=5, description="Resolution, composition, professional quality")
    relevance_score: int = Field(ge=1, le=5, description="Thematic connection to the article")
    reasoning: str = Field(description="Brief explanation of selection")


SELECTION_SYSTEM = """You are a magazine photo editor selecting images for a long-form article.

Score each image on these criteria (1-5 each):

1. **Brief Compliance** (MOST IMPORTANT — 40% weight):
   - Does the image match what the brief EXPLICITLY requests?
   - If the brief says "metaphorical" or "evocative," reject literal depictions.
   - If the brief says to AVOID certain imagery, any match is an automatic 1.

2. **Mood/Tone Match** (25% weight):
   - Does the image's mood match the brief's tone requirements?

3. **Visual Quality** (20% weight):
   - Resolution, composition, professional quality.

4. **Article Relevance** (15% weight — LOWEST):
   - Thematic connection to the article.
   - A beautiful, mood-appropriate image loosely connected to the topic
     is BETTER than an ugly, literal depiction.

Select the image with the highest weighted total."""

SELECTION_USER = """Select the best image for the given context.

Context: {context}

Search query: {query}

Image candidates:
{candidates}

Choose the image index (0-based) that best matches the context.
Score the WINNING image on all four criteria."""

SELECTION_USER_WITH_CRITERIA = """Select the best image based on the following criteria.

**Selection Criteria:**
{criteria}

**Document Context:**
{context}

**Search query:** {query}

**Image candidates:**
{candidates}

Choose the image index (0-based) that best matches the criteria.
Score the WINNING image on all four criteria."""


async def select_best_image(
    candidates: list[ImageResult],
    query: str,
    context: str,
    custom_selection_criteria: str | None = None,
) -> ImageResult:
    """Use LLM to select best image from candidates.

    Args:
        candidates: List of image results to choose from
        query: Original search query
        context: Article/document context
        custom_selection_criteria: If provided, use these detailed criteria
            for selection instead of the generic selection prompt. Useful
            for more specific matching requirements.

    Returns:
        Best matching ImageResult
    """
    if len(candidates) <= 1:
        return candidates[0] if candidates else candidates[0]

    # Format candidates for LLM
    candidate_text = "\n".join(
        f"{i}. [{c.provider.value}] "
        f"{c.metadata.alt_text or c.metadata.description or 'No description'} "
        f"({c.metadata.width}x{c.metadata.height})"
        for i, c in enumerate(candidates)
    )

    try:
        # Use custom criteria prompt if provided
        if custom_selection_criteria:
            user_prompt = SELECTION_USER_WITH_CRITERIA.format(
                criteria=custom_selection_criteria,
                context=context[:1500],  # Less context since criteria is detailed
                query=query,
                candidates=candidate_text,
            )
        else:
            user_prompt = SELECTION_USER.format(
                context=context[:2000],
                query=query,
                candidates=candidate_text,
            )

        result = await invoke(
            tier=ModelTier.HAIKU,
            system=SELECTION_SYSTEM,
            user=user_prompt,
            schema=ImageSelection,
        )

        index = max(0, min(result.selected_index, len(candidates) - 1))
        logger.info(
            f"LLM selected image {index}: "
            f"brief={result.brief_compliance_score} mood={result.mood_score} "
            f"quality={result.quality_score} relevance={result.relevance_score} "
            f"| {result.reasoning}"
        )
        return candidates[index]

    except Exception as e:
        logger.warning(f"LLM selection failed, using first result: {e}")
        return candidates[0]
