"""Unified structured output interface for LLM responses.

Provides a single entry point that automatically selects the optimal strategy
for obtaining structured output from Claude models, with consistent error handling
and retry logic across all patterns.

Strategies (auto-selected based on context):
- LANGCHAIN_STRUCTURED: LangChain's .with_structured_output() (default for single requests)
- BATCH_TOOL_CALL: Anthropic Batch API with tool calling (when prefer_batch_api=True)
- TOOL_AGENT: Multi-turn tool agent with output as final tool (when tools provided)
- JSON_PROMPTING: Fallback JSON extraction from response (for edge cases)

Example - Single request:
    result = await get_structured_output(
        output_schema=PaperAnalysis,
        user_prompt="Analyze this paper",
        system_prompt=ANALYZER_SYSTEM,
        tier=ModelTier.SONNET,
    )

Example - Batch request (auto-selects batch API for cost savings):
    results = await get_structured_output(
        output_schema=PaperSummary,
        requests=[
            StructuredRequest(id="paper-1", user_prompt="Summarize: ..."),
            StructuredRequest(id="paper-2", user_prompt="Summarize: ..."),
        ],
        system_prompt=SUMMARIZER_SYSTEM,
        tier=ModelTier.HAIKU,
    )

Example - With tools (multi-turn agent):
    result = await get_structured_output(
        output_schema=ResearchOutput,
        user_prompt="Research the topic...",
        system_prompt=RESEARCHER_SYSTEM,
        tools=[search_tool, fetch_tool],
        tier=ModelTier.SONNET,
    )
"""

from .convenience import (
    classify_content,
    extract_from_text,
    get_structured_output_with_result,
)
from .interface import get_structured_output
from .types import (
    BatchResult,
    StructuredOutputConfig,
    StructuredOutputError,
    StructuredOutputResult,
    StructuredOutputStrategy,
    StructuredRequest,
)

__all__ = [
    "StructuredOutputStrategy",
    "StructuredOutputConfig",
    "StructuredRequest",
    "StructuredOutputResult",
    "BatchResult",
    "StructuredOutputError",
    "get_structured_output",
    "get_structured_output_with_result",
    "extract_from_text",
    "classify_content",
]
