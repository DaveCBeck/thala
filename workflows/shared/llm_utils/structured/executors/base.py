"""Abstract base class for structured output strategy executors."""

import json
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Type, TypeVar, Union, get_origin

from pydantic import BaseModel

from ..types import StructuredOutputConfig, StructuredOutputResult

T = TypeVar("T", bound=BaseModel)

# Type alias for user content - either string or multimodal content blocks
UserContent = Union[str, list[dict[str, Any]]]


def coerce_to_schema(data: dict[str, Any], schema: Type[BaseModel]) -> dict[str, Any]:
    """Coerce parsed JSON to match schema types before Pydantic validation.

    Handles common LLM output issues:
    - Strings returned for list fields -> converted to single-item lists
    - Empty strings for list fields -> converted to empty lists

    Args:
        data: Parsed JSON dict from LLM response
        schema: Pydantic model class to validate against

    Returns:
        Coerced dict ready for model_validate()
    """
    if not isinstance(data, dict):
        return data

    coerced = data.copy()

    for field_name, field_info in schema.model_fields.items():
        if field_name not in coerced:
            continue

        annotation = field_info.annotation
        origin = get_origin(annotation)
        value = coerced[field_name]

        # Handle list fields where LLM returned a string
        if origin is list and isinstance(value, str):
            if not value.strip():
                coerced[field_name] = []
            else:
                # Try parsing as JSON array first (LLM sometimes stringifies lists)
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        coerced[field_name] = parsed
                    else:
                        coerced[field_name] = [value]
                except (json.JSONDecodeError, ValueError):
                    coerced[field_name] = [value]

    return coerced


class StrategyExecutor(ABC, Generic[T]):
    """Abstract base for structured output strategies."""

    @abstractmethod
    async def execute(
        self,
        output_schema: Type[T],
        user_prompt: UserContent,
        system_prompt: Optional[str],
        output_config: StructuredOutputConfig,
    ) -> StructuredOutputResult[T]:
        """Execute the strategy and return result.

        Args:
            output_schema: Pydantic model class for structured output
            user_prompt: Either a string or multimodal content blocks
                (list of dicts with "type" key, e.g. [{"type": "text", ...}, {"type": "image", ...}])
            system_prompt: Optional system prompt
            output_config: Configuration for the structured output operation

        Note: Parameter is named 'output_config' (not 'config') to avoid conflict
        with LangSmith's @traceable decorator which treats 'config' as a LangChain
        RunnableConfig dict and tries to call .get("callbacks") on it.
        """
        pass


__all__ = ["StrategyExecutor", "coerce_to_schema", "UserContent"]
