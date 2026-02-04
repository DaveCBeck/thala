"""Convenience functions for common structured output tasks."""

from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from ..models import ModelTier
from .executors import executors
from .interface import get_structured_output
from .retry import with_retries
from .strategy_selection import select_strategy
from .types import StructuredOutputConfig, StructuredOutputResult

T = TypeVar("T", bound=BaseModel)


async def get_structured_output_with_result(
    output_schema: Type[T],
    user_prompt: str,
    system_prompt: Optional[str] = None,
    config: Optional[StructuredOutputConfig] = None,
    **kwargs,
) -> StructuredOutputResult[T]:
    """Same as get_structured_output but returns full result with metadata.

    Use this when you need access to:
    - Which strategy was used
    - Token usage information
    - Extended thinking content
    - Error details without exceptions

    Returns:
        StructuredOutputResult containing value or error (never raises)
    """
    effective_config = config or StructuredOutputConfig()
    for key, value in kwargs.items():
        if value is not None and hasattr(effective_config, key):
            setattr(effective_config, key, value)

    selected_strategy = select_strategy(effective_config, is_batch=False, batch_size=0)
    executor = executors[selected_strategy]

    async def _invoke() -> StructuredOutputResult[T]:
        return await executor.execute(
            output_schema=output_schema,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            config=effective_config,
        )

    return await with_retries(_invoke, effective_config, output_schema, selected_strategy)


async def extract_from_text(
    output_schema: Type[T],
    text: str,
    extraction_prompt: str,
    tier: ModelTier = ModelTier.SONNET,
    **kwargs,
) -> T:
    """Convenience wrapper for text extraction tasks.

    Example:
        metadata = await extract_from_text(
            output_schema=PaperMetadata,
            text=paper_content,
            extraction_prompt="Extract paper metadata including title, authors, and abstract",
        )
    """
    return await get_structured_output(
        output_schema=output_schema,
        user_prompt=f"{extraction_prompt}\n\nText:\n{text}",
        tier=tier,
        **kwargs,
    )


async def classify_content(
    content: str,
    classification_schema: Type[T],
    instructions: str,
    tier: ModelTier = ModelTier.HAIKU,
    **kwargs,
) -> T:
    """Convenience wrapper for classification tasks.

    Example:
        result = await classify_content(
            content=scraped_html,
            classification_schema=ContentClassification,
            instructions="Classify the content type as: full_text, abstract, or paywall",
            tier=ModelTier.HAIKU,
        )
    """
    return await get_structured_output(
        output_schema=classification_schema,
        user_prompt=content,
        system_prompt=instructions,
        tier=tier,
        **kwargs,
    )


__all__ = [
    "get_structured_output_with_result",
    "extract_from_text",
    "classify_content",
]
