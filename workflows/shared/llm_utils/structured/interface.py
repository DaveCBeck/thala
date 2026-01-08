"""Unified structured output interface.

Provides the main get_structured_output() function that automatically selects
the optimal strategy for obtaining structured output from Claude models.

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

import logging
from typing import Callable, Optional, Type, TypeVar, Union, overload

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ..models import ModelTier
from .batch_executor import execute_batch_concurrent
from .executors import executors
from .executors.batch import BatchToolCallExecutor
from .retry import with_retries
from .strategy_selection import select_strategy
from .types import (
    BatchResult,
    StructuredOutputConfig,
    StructuredOutputError,
    StructuredOutputResult,
    StructuredOutputStrategy,
    StructuredRequest,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@overload
async def get_structured_output(
    output_schema: Type[T],
    *,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    config: Optional[StructuredOutputConfig] = None,
    tier: Optional[ModelTier] = None,
    max_tokens: Optional[int] = None,
    thinking_budget: Optional[int] = None,
    strategy: Optional[StructuredOutputStrategy] = None,
    use_json_schema_method: Optional[bool] = None,
    prefer_batch_api: Optional[bool] = None,
    max_retries: Optional[int] = None,
    enable_prompt_cache: Optional[bool] = None,
    tools: Optional[list[BaseTool]] = None,
    max_tool_calls: Optional[int] = None,
) -> T: ...


@overload
async def get_structured_output(
    output_schema: Type[T],
    *,
    requests: list[StructuredRequest],
    system_prompt: Optional[str] = None,
    config: Optional[StructuredOutputConfig] = None,
    tier: Optional[ModelTier] = None,
    max_tokens: Optional[int] = None,
    thinking_budget: Optional[int] = None,
    strategy: Optional[StructuredOutputStrategy] = None,
    use_json_schema_method: Optional[bool] = None,
    prefer_batch_api: Optional[bool] = None,
    batch_threshold: Optional[int] = None,
    max_retries: Optional[int] = None,
    enable_prompt_cache: Optional[bool] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> BatchResult[T]: ...


async def get_structured_output(
    output_schema: Type[T],
    *,
    # Single request params
    user_prompt: Optional[str] = None,
    # Batch request params
    requests: Optional[list[StructuredRequest]] = None,
    # Common params
    system_prompt: Optional[str] = None,
    config: Optional[StructuredOutputConfig] = None,
    # Config overrides (convenience)
    tier: Optional[ModelTier] = None,
    max_tokens: Optional[int] = None,
    thinking_budget: Optional[int] = None,
    strategy: Optional[StructuredOutputStrategy] = None,
    use_json_schema_method: Optional[bool] = None,
    prefer_batch_api: Optional[bool] = None,
    batch_threshold: Optional[int] = None,
    max_retries: Optional[int] = None,
    enable_prompt_cache: Optional[bool] = None,
    tools: Optional[list[BaseTool]] = None,
    max_tool_calls: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Union[T, BatchResult[T]]:
    """Get structured output from LLM, automatically selecting the best strategy.

    This is the unified interface for both single and batch structured output requests.
    The function determines whether to process as single or batch based on which
    parameters are provided:

    - If `user_prompt` is provided: Single request, returns T
    - If `requests` is provided: Batch request, returns BatchResult[T]

    The strategy is auto-selected based on context:
    - Tools provided -> TOOL_AGENT (multi-turn with tool use)
    - prefer_batch_api=True -> BATCH_TOOL_CALL (50% cost reduction)
    - Batch with 5+ items -> BATCH_TOOL_CALL (50% cost reduction)
    - Otherwise -> LANGCHAIN_STRUCTURED (default)

    Args:
        output_schema: Pydantic model class for the expected output

        # Single request (provide one of these):
        user_prompt: User message content for single request

        # Batch request (provide this instead):
        requests: List of StructuredRequest for batch processing

        # Common parameters:
        system_prompt: System message (default for all batch items, or for single)
        config: Full configuration object (alternative to individual params)

        # Config overrides (convenience - override specific config fields):
        tier: Model tier (HAIKU, SONNET, SONNET_1M, OPUS)
        max_tokens: Maximum output tokens
        thinking_budget: Token budget for extended thinking
        strategy: Force a specific strategy
        use_json_schema_method: Use stricter JSON schema validation
        prefer_batch_api: Route ALL requests through batch API for 50% cost savings.
            Set to True for cost optimization when latency isn't a concern.
            Defaults to THALA_PREFER_BATCH_API env var.
        batch_threshold: Min items to use batch API when prefer_batch_api=False (default: 5)
        max_retries: Retry attempts on failure
        enable_prompt_cache: Enable prompt caching
        tools: LangChain tools (triggers TOOL_AGENT strategy)
        max_tool_calls: Max tool calls for agent
        progress_callback: Callback(completed, total) for batch progress

    Returns:
        - Single request: Validated Pydantic model instance (T)
        - Batch request: BatchResult[T] with results dict

    Raises:
        ValueError: If neither user_prompt nor requests is provided
        StructuredOutputError: If extraction fails after retries

    Examples:
        # Single request
        analysis = await get_structured_output(
            output_schema=PaperAnalysis,
            user_prompt="Analyze this paper: ...",
            system_prompt=ANALYZER_SYSTEM,
            tier=ModelTier.SONNET,
        )

        # Batch request (auto-uses batch API for 5+ items)
        results = await get_structured_output(
            output_schema=PaperSummary,
            requests=[
                StructuredRequest(id="p1", user_prompt="Summarize: ..."),
                StructuredRequest(id="p2", user_prompt="Summarize: ..."),
            ],
            system_prompt=SUMMARIZER_SYSTEM,
            tier=ModelTier.HAIKU,
        )
        for id, result in results.results.items():
            if result.success:
                print(result.value.summary)

        # With tools (multi-turn agent)
        research = await get_structured_output(
            output_schema=ResearchOutput,
            user_prompt="Research the topic...",
            system_prompt=RESEARCHER_SYSTEM,
            tools=[search_tool, fetch_tool],
        )
    """
    # Validate input
    if user_prompt is None and requests is None:
        raise ValueError("Must provide either user_prompt (single) or requests (batch)")
    if user_prompt is not None and requests is not None:
        raise ValueError("Cannot provide both user_prompt and requests")

    # Build effective config
    effective_config = config or StructuredOutputConfig()
    if tier is not None:
        effective_config.tier = tier
    if max_tokens is not None:
        effective_config.max_tokens = max_tokens
    if thinking_budget is not None:
        effective_config.thinking_budget = thinking_budget
    if strategy is not None:
        effective_config.strategy = strategy
    if use_json_schema_method is not None:
        effective_config.use_json_schema_method = use_json_schema_method
    if prefer_batch_api is not None:
        effective_config.prefer_batch_api = prefer_batch_api
    if batch_threshold is not None:
        effective_config.batch_threshold = batch_threshold
    if max_retries is not None:
        effective_config.max_retries = max_retries
    if enable_prompt_cache is not None:
        effective_config.enable_prompt_cache = enable_prompt_cache
    if tools is not None:
        effective_config.tools = tools
    if max_tool_calls is not None:
        effective_config.max_tool_calls = max_tool_calls

    # Determine if batch
    is_batch = requests is not None
    batch_size = len(requests) if requests else 0

    # Select strategy
    selected_strategy = select_strategy(effective_config, is_batch, batch_size)
    logger.debug(f"Selected strategy: {selected_strategy.name}")

    # Execute
    if is_batch:
        return await _execute_batch(
            output_schema=output_schema,
            requests=requests,
            system_prompt=system_prompt,
            config=effective_config,
            selected_strategy=selected_strategy,
            progress_callback=progress_callback,
        )
    else:
        return await _execute_single(
            output_schema=output_schema,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            config=effective_config,
            selected_strategy=selected_strategy,
        )


async def _execute_single(
    output_schema: Type[T],
    user_prompt: str,
    system_prompt: Optional[str],
    config: StructuredOutputConfig,
    selected_strategy: StructuredOutputStrategy,
) -> T:
    """Execute single request.

    When selected_strategy is BATCH_TOOL_CALL (e.g., prefer_batch_api=True),
    wraps the single request as a batch for 50% cost savings.
    """
    # For BATCH_TOOL_CALL strategy, wrap single request as batch
    if selected_strategy == StructuredOutputStrategy.BATCH_TOOL_CALL:
        executor = BatchToolCallExecutor()

        async def _invoke() -> StructuredOutputResult[T]:
            results = await executor.execute_batch(
                output_schema=output_schema,
                requests=[StructuredRequest(id="_single", user_prompt=user_prompt)],
                default_system=system_prompt,
                config=config,
            )
            return results["_single"]

    else:
        executor = executors[selected_strategy]

        async def _invoke() -> StructuredOutputResult[T]:
            return await executor.execute(
                output_schema=output_schema,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                config=config,
            )

    result = await with_retries(_invoke, config, output_schema, selected_strategy)

    if not result.success:
        raise StructuredOutputError(
            message=result.error or "Unknown error",
            schema=output_schema,
            strategy=selected_strategy,
            attempts=config.max_retries,
        )

    return result.value


async def _execute_batch(
    output_schema: Type[T],
    requests: list[StructuredRequest],
    system_prompt: Optional[str],
    config: StructuredOutputConfig,
    selected_strategy: StructuredOutputStrategy,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> BatchResult[T]:
    """Execute batch request."""
    # For batch, we can use BATCH_TOOL_CALL or fall back to concurrent single requests
    if selected_strategy == StructuredOutputStrategy.BATCH_TOOL_CALL:
        executor = BatchToolCallExecutor()
        results = await executor.execute_batch(
            output_schema=output_schema,
            requests=requests,
            default_system=system_prompt,
            config=config,
            progress_callback=progress_callback,
        )
    else:
        # Fallback: concurrent single requests
        results = await execute_batch_concurrent(
            output_schema=output_schema,
            requests=requests,
            system_prompt=system_prompt,
            config=config,
            selected_strategy=selected_strategy,
            progress_callback=progress_callback,
        )

    # Build BatchResult
    successful = sum(1 for r in results.values() if r.success)
    failed = len(results) - successful

    return BatchResult(
        results=results,
        total_items=len(requests),
        successful_items=successful,
        failed_items=failed,
        strategy_used=selected_strategy,
    )


__all__ = [
    "get_structured_output",
]
