"""Optional LLM-based image selection."""

import logging

from pydantic import BaseModel, Field

from workflows.shared.llm_utils import ModelTier, get_structured_output

from .types import ImageResult

logger = logging.getLogger(__name__)


class ImageSelection(BaseModel):
    """LLM output for image selection."""

    selected_index: int = Field(description="0-based index of best image")
    reasoning: str = Field(description="Brief explanation of selection")


SELECTION_SYSTEM = """You are an image selector for article headers.
Select the image that best complements the article context without being too literal.
Consider composition, mood, and thematic relevance."""

SELECTION_USER = """Select the best image for the given context.

Context: {context}

Search query: {query}

Image candidates:
{candidates}

Choose the image index (0-based) that best matches the context."""


async def select_best_image(
    candidates: list[ImageResult],
    query: str,
    context: str,
) -> ImageResult:
    """Use LLM to select best image from candidates.

    Args:
        candidates: List of image results to choose from
        query: Original search query
        context: Article/document context

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
        result = await get_structured_output(
            output_schema=ImageSelection,
            user_prompt=SELECTION_USER.format(
                context=context[:2000],  # Truncate long context
                query=query,
                candidates=candidate_text,
            ),
            system_prompt=SELECTION_SYSTEM,
            tier=ModelTier.HAIKU,  # Cheapest tier for simple selection
        )

        index = max(0, min(result.selected_index, len(candidates) - 1))
        logger.debug(f"LLM selected image {index}: {result.reasoning}")
        return candidates[index]

    except Exception as e:
        logger.warning(f"LLM selection failed, using first result: {e}")
        return candidates[0]
