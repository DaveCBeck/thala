"""Abstract base class for structured output strategy executors."""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Type, TypeVar, get_origin

from pydantic import BaseModel

from ..types import StructuredOutputConfig, StructuredOutputResult

T = TypeVar("T", bound=BaseModel)


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
            # Empty or whitespace-only string -> empty list
            if not value.strip():
                coerced[field_name] = []
            else:
                # Non-empty string -> single-item list
                coerced[field_name] = [value]

    return coerced


class StrategyExecutor(ABC, Generic[T]):
    """Abstract base for structured output strategies."""

    @abstractmethod
    async def execute(
        self,
        output_schema: Type[T],
        user_prompt: str,
        system_prompt: Optional[str],
        config: StructuredOutputConfig,
    ) -> StructuredOutputResult[T]:
        """Execute the strategy and return result."""
        pass


__all__ = ["StrategyExecutor", "coerce_to_schema"]
