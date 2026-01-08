"""Abstract base class for structured output strategy executors."""

from abc import ABC, abstractmethod
from typing import Generic, Optional, Type, TypeVar

from pydantic import BaseModel

from ..types import StructuredOutputConfig, StructuredOutputResult

T = TypeVar("T", bound=BaseModel)


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


__all__ = ["StrategyExecutor"]
