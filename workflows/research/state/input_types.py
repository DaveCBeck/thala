"""Input types for research workflow."""

from typing import Literal, Optional
from typing_extensions import TypedDict


class ResearchInput(TypedDict):
    """Initial user research request."""

    query: str  # Original user query
    depth: Literal["quick", "standard", "comprehensive"]
    max_sources: int  # Max web sources to use
    max_iterations: Optional[int]  # Override default for depth

    # Language configuration
    language: Optional[str]  # Single language mode: ISO 639-1 code (e.g., "es", "zh")
    translate_to: Optional[str]  # Translate final output to this language
    preserve_quotes: Optional[bool]  # Keep quotes in original language when translating


class ClarificationQuestion(TypedDict):
    """Question to clarify user intent."""

    question: str
    options: Optional[list[str]]  # Suggested answers (if applicable)


class ResearchBrief(TypedDict):
    """Refined research brief after clarification."""

    topic: str  # Core research topic
    objectives: list[str]  # Specific research objectives
    scope: str  # What's in/out of scope
    key_questions: list[str]  # Questions to answer
    memory_context: str  # Relevant memory findings summary
