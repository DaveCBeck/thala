"""
Tests for the unified invoke() function and InvokeConfig.

Tests cover:
- InvokeConfig validation
- Routing paths (DeepSeek, broker, direct)
- Batch input handling
- invoke_batch() context manager
- Broker response conversion
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from workflows.shared.llm_utils.config import InvokeConfig
from workflows.shared.llm_utils.invoke import (
    InvokeBatch,
    _broker_response_to_message,
    _invoke_direct,
    _invoke_via_broker,
    invoke,
    invoke_batch,
)
from workflows.shared.llm_utils.models import ModelTier


class TestInvokeConfig:
    """Tests for InvokeConfig dataclass validation."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = InvokeConfig()
        assert config.cache is True
        assert config.cache_ttl == "5m"
        assert config.batch_policy is None
        assert config.thinking_budget is None
        assert config.max_tokens == 4096

    def test_cache_with_thinking_budget_allowed_in_config(self):
        """InvokeConfig allows cache+thinking_budget; validation deferred to invoke().

        This is because DeepSeek R1 supports cache+thinking (automatic prefix caching
        is independent of thinking). The validation is tier-specific in invoke().
        """
        config = InvokeConfig(cache=True, thinking_budget=8000)
        assert config.cache is True
        assert config.thinking_budget == 8000

    @pytest.mark.asyncio
    async def test_anthropic_cache_with_thinking_budget_raises_in_invoke(self):
        """For Anthropic models, cache+thinking_budget raises ValueError in invoke()."""
        with pytest.raises(ValueError, match="Cannot use cache with extended thinking"):
            await invoke(
                tier=ModelTier.OPUS,
                system="Test",
                user="Test",
                config=InvokeConfig(cache=True, thinking_budget=8000),
            )

    @pytest.mark.asyncio
    async def test_deepseek_cache_with_thinking_budget_allowed(self):
        """DeepSeek allows cache+thinking_budget (automatic prefix caching)."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Response")

        with patch("workflows.shared.llm_utils.invoke.get_llm", return_value=mock_llm):
            # Should NOT raise - DeepSeek supports this combination
            result = await invoke(
                tier=ModelTier.DEEPSEEK_R1,
                system="Test",
                user="Test",
                config=InvokeConfig(cache=True, thinking_budget=8000),
            )
            assert result.content == "Response"

    def test_thinking_budget_without_cache_ok(self):
        """thinking_budget with cache=False is valid."""
        config = InvokeConfig(cache=False, thinking_budget=8000)
        assert config.thinking_budget == 8000
        assert config.cache is False

    def test_batch_policy_accepted(self):
        """batch_policy can be set."""
        from core.llm_broker import BatchPolicy

        config = InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE)
        assert config.batch_policy == BatchPolicy.PREFER_BALANCE

    def test_tools_config(self):
        """Tools can be configured."""
        tools = [{"name": "search", "type": "function"}]
        config = InvokeConfig(tools=tools, tool_choice={"type": "auto"})
        assert config.tools == tools
        assert config.tool_choice == {"type": "auto"}


class TestBrokerResponseToMessage:
    """Tests for _broker_response_to_message conversion."""

    def test_basic_conversion(self):
        """Basic response conversion preserves content."""
        from core.llm_broker import LLMResponse

        response = LLMResponse(
            request_id="test-123",
            content="Hello, world!",
            success=True,
            usage={"input_tokens": 10, "output_tokens": 5},
            model="claude-sonnet-4-5-20250929",
            stop_reason="end_turn",
        )

        message = _broker_response_to_message(response)

        assert isinstance(message, AIMessage)
        assert message.content == "Hello, world!"
        assert message.response_metadata["usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert message.response_metadata["model"] == "claude-sonnet-4-5-20250929"
        assert message.response_metadata["stop_reason"] == "end_turn"

    def test_with_thinking(self):
        """Response with thinking content includes it in additional_kwargs."""
        from core.llm_broker import LLMResponse

        response = LLMResponse(
            request_id="test-456",
            content="The answer is 42.",
            success=True,
            thinking="Let me think about this carefully...",
        )

        message = _broker_response_to_message(response)

        assert message.additional_kwargs["thinking"] == "Let me think about this carefully..."

    def test_without_thinking(self):
        """Response without thinking has empty additional_kwargs."""
        from core.llm_broker import LLMResponse

        response = LLMResponse(
            request_id="test-789",
            content="Simple response.",
            success=True,
        )

        message = _broker_response_to_message(response)

        assert "thinking" not in message.additional_kwargs


class TestInvokeRouting:
    """Tests for invoke() routing logic."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns AIMessage."""
        mock = AsyncMock()
        mock.ainvoke.return_value = AIMessage(content="Mock response")
        return mock

    @pytest.mark.asyncio
    async def test_deepseek_routes_direct(self, mock_llm):
        """DeepSeek tiers should route directly, not through broker."""
        with patch("workflows.shared.llm_utils.invoke.get_llm", return_value=mock_llm) as mock_get_llm:
            result = await invoke(
                tier=ModelTier.DEEPSEEK_V3,
                system="Test system",
                user="Test user",
            )

        mock_get_llm.assert_called_once()
        assert mock_llm.ainvoke.called
        assert result.content == "Mock response"

    @pytest.mark.asyncio
    async def test_deepseek_ignores_batch_policy(self, mock_llm):
        """DeepSeek should ignore batch_policy and route directly."""
        from core.llm_broker import BatchPolicy

        with (
            patch("workflows.shared.llm_utils.invoke.get_llm", return_value=mock_llm),
            patch("core.llm_broker.is_broker_enabled", return_value=True),
            patch("core.llm_broker.get_broker") as mock_broker,
        ):
            result = await invoke(
                tier=ModelTier.DEEPSEEK_R1,
                system="Test",
                user="Test",
                config=InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE, cache=False),
            )

        # Broker should NOT be called for DeepSeek
        mock_broker.assert_not_called()
        assert result.content == "Mock response"

    @pytest.mark.asyncio
    async def test_anthropic_with_batch_policy_routes_to_broker(self):
        """Anthropic with batch_policy routes through broker when enabled."""
        from core.llm_broker import BatchPolicy, LLMResponse

        mock_future = asyncio.Future()
        mock_future.set_result(
            LLMResponse(
                request_id="test",
                content="Broker response",
                success=True,
                usage={"input_tokens": 10, "output_tokens": 5},
            )
        )

        mock_broker = MagicMock()
        mock_broker.batch_group.return_value.__aenter__ = AsyncMock()
        mock_broker.batch_group.return_value.__aexit__ = AsyncMock()
        mock_broker.request = AsyncMock(return_value=mock_future)

        with (
            patch("core.llm_broker.is_broker_enabled", return_value=True),
            patch("core.llm_broker.get_broker", return_value=mock_broker),
        ):
            result = await invoke(
                tier=ModelTier.SONNET,
                system="Test system",
                user="Test user",
                config=InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE),
            )

        assert mock_broker.request.called
        assert result.content == "Broker response"

    @pytest.mark.asyncio
    async def test_anthropic_without_batch_policy_routes_direct(self, mock_llm):
        """Anthropic without batch_policy routes directly."""
        with patch("workflows.shared.llm_utils.invoke.get_llm", return_value=mock_llm):
            result = await invoke(
                tier=ModelTier.SONNET,
                system="Test system",
                user="Test user",
            )

        assert mock_llm.ainvoke.called
        assert result.content == "Mock response"

    @pytest.mark.asyncio
    async def test_broker_disabled_falls_back_to_direct(self, mock_llm):
        """When broker is disabled, falls back to direct invocation."""
        from core.llm_broker import BatchPolicy

        with (
            patch("workflows.shared.llm_utils.invoke.get_llm", return_value=mock_llm),
            patch("core.llm_broker.is_broker_enabled", return_value=False),
        ):
            result = await invoke(
                tier=ModelTier.HAIKU,
                system="Test",
                user="Test",
                config=InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE),
            )

        assert mock_llm.ainvoke.called
        assert result.content == "Mock response"


class TestInvokeBatchInput:
    """Tests for invoke() batch input handling."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns different responses for each call."""
        mock = AsyncMock()
        mock.ainvoke.side_effect = [
            AIMessage(content="Response 1"),
            AIMessage(content="Response 2"),
            AIMessage(content="Response 3"),
        ]
        return mock

    @pytest.mark.asyncio
    async def test_list_input_returns_list(self, mock_llm):
        """List input should return list of responses."""
        with patch("workflows.shared.llm_utils.invoke.get_llm", return_value=mock_llm):
            results = await invoke(
                tier=ModelTier.HAIKU,
                system="Summarize",
                user=["Doc 1", "Doc 2", "Doc 3"],
            )

        assert isinstance(results, list)
        assert len(results) == 3
        assert results[0].content == "Response 1"
        assert results[1].content == "Response 2"
        assert results[2].content == "Response 3"

    @pytest.mark.asyncio
    async def test_single_input_returns_single(self, mock_llm):
        """Single string input should return single response."""
        mock_llm.ainvoke.side_effect = None
        mock_llm.ainvoke.return_value = AIMessage(content="Single response")

        with patch("workflows.shared.llm_utils.invoke.get_llm", return_value=mock_llm):
            result = await invoke(
                tier=ModelTier.HAIKU,
                system="Test",
                user="Single prompt",
            )

        assert isinstance(result, AIMessage)
        assert result.content == "Single response"


class TestInvokeDirect:
    """Tests for _invoke_direct function."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = AsyncMock()
        mock.ainvoke.return_value = AIMessage(content="Direct response")
        return mock

    @pytest.mark.asyncio
    async def test_anthropic_with_cache(self, mock_llm):
        """Anthropic with cache=True uses cached messages."""
        with (
            patch("workflows.shared.llm_utils.invoke.get_llm", return_value=mock_llm),
            patch("workflows.shared.llm_utils.invoke.create_cached_messages") as mock_cache,
        ):
            mock_cache.return_value = [{"role": "system", "content": "cached"}]

            config = InvokeConfig(cache=True)
            await _invoke_direct(ModelTier.SONNET, "System", ["User"], config)

            mock_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_anthropic_without_cache(self, mock_llm):
        """Anthropic with cache=False uses plain messages."""
        with (
            patch("workflows.shared.llm_utils.invoke.get_llm", return_value=mock_llm),
            patch("workflows.shared.llm_utils.invoke.create_cached_messages") as mock_cache,
        ):
            config = InvokeConfig(cache=False)
            await _invoke_direct(ModelTier.SONNET, "System", ["User"], config)

            mock_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_deepseek_never_uses_cache_control(self, mock_llm):
        """DeepSeek should not use Anthropic-style cache control."""
        with (
            patch("workflows.shared.llm_utils.invoke.get_llm", return_value=mock_llm),
            patch("workflows.shared.llm_utils.invoke.create_cached_messages") as mock_cache,
        ):
            config = InvokeConfig(cache=True)  # Should be ignored for DeepSeek
            await _invoke_direct(ModelTier.DEEPSEEK_V3, "System", ["User"], config)

            mock_cache.assert_not_called()


class TestInvokeViaBroker:
    """Tests for _invoke_via_broker function."""

    @pytest.mark.asyncio
    async def test_submits_within_batch_group(self):
        """All requests should be submitted within batch_group context."""
        from core.llm_broker import BatchPolicy, LLMResponse

        futures = [asyncio.Future() for _ in range(2)]
        for i, f in enumerate(futures):
            f.set_result(
                LLMResponse(
                    request_id=f"test-{i}",
                    content=f"Response {i}",
                    success=True,
                )
            )

        mock_broker = MagicMock()
        mock_broker.batch_group.return_value.__aenter__ = AsyncMock()
        mock_broker.batch_group.return_value.__aexit__ = AsyncMock()
        mock_broker.request = AsyncMock(side_effect=futures)

        with patch("core.llm_broker.get_broker", return_value=mock_broker):
            config = InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE)
            results = await _invoke_via_broker(ModelTier.SONNET, "System", ["User 1", "User 2"], config)

        assert mock_broker.batch_group.called
        assert mock_broker.request.call_count == 2
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_raises_on_broker_failure(self):
        """Should raise RuntimeError on broker failure."""
        from core.llm_broker import BatchPolicy, LLMResponse

        mock_future = asyncio.Future()
        mock_future.set_result(
            LLMResponse(
                request_id="test",
                content="",
                success=False,
                error="API rate limited",
            )
        )

        mock_broker = MagicMock()
        mock_broker.batch_group.return_value.__aenter__ = AsyncMock()
        mock_broker.batch_group.return_value.__aexit__ = AsyncMock()
        mock_broker.request = AsyncMock(return_value=mock_future)

        with patch("core.llm_broker.get_broker", return_value=mock_broker):
            config = InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE)

            with pytest.raises(RuntimeError, match="Broker request failed"):
                await _invoke_via_broker(ModelTier.SONNET, "System", ["User"], config)


class TestInvokeStructuredOutput:
    """Tests for invoke() with schema= parameter."""

    @pytest.mark.asyncio
    async def test_schema_returns_pydantic_model(self):
        """When schema is provided, should return validated Pydantic model."""
        from pydantic import BaseModel
        from workflows.shared.llm_utils.structured.types import StructuredOutputResult

        class TestOutput(BaseModel):
            value: str
            score: float

        mock_result = StructuredOutputResult.ok(
            value=TestOutput(value="test", score=0.9),
            strategy=MagicMock(),
        )

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=mock_result)

        with (
            patch("workflows.shared.llm_utils.structured.executors.get_executor", return_value=mock_executor),
            patch("workflows.shared.llm_utils.structured.retry.with_retries", return_value=mock_result),
        ):
            result = await invoke(
                tier=ModelTier.SONNET,
                system="Extract data.",
                user="Some content",
                schema=TestOutput,
            )

        assert isinstance(result, TestOutput)
        assert result.value == "test"
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_schema_with_list_input_returns_list(self):
        """When schema and list input provided, should return list of models."""
        from pydantic import BaseModel
        from workflows.shared.llm_utils.structured.types import StructuredOutputResult

        class TestOutput(BaseModel):
            value: str

        mock_results = [
            StructuredOutputResult.ok(value=TestOutput(value="a"), strategy=MagicMock()),
            StructuredOutputResult.ok(value=TestOutput(value="b"), strategy=MagicMock()),
        ]

        call_count = 0

        async def mock_with_retries_impl(*args, **kwargs):
            nonlocal call_count
            result = mock_results[call_count]
            call_count += 1
            return result

        mock_executor = MagicMock()

        with (
            patch("workflows.shared.llm_utils.structured.executors.get_executor", return_value=mock_executor),
            patch("workflows.shared.llm_utils.structured.retry.with_retries", side_effect=mock_with_retries_impl),
        ):
            results = await invoke(
                tier=ModelTier.HAIKU,
                system="Extract data.",
                user=["Content 1", "Content 2"],
                schema=TestOutput,
            )

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].value == "a"
        assert results[1].value == "b"

    @pytest.mark.asyncio
    async def test_schema_with_tools_uses_tool_agent(self):
        """When schema and tools provided, should use TOOL_AGENT strategy."""
        from pydantic import BaseModel
        from workflows.shared.llm_utils.structured.types import (
            StructuredOutputResult,
            StructuredOutputStrategy,
        )

        class TestOutput(BaseModel):
            value: str

        mock_result = StructuredOutputResult.ok(
            value=TestOutput(value="tool result"),
            strategy=StructuredOutputStrategy.TOOL_AGENT,
        )

        captured_strategy = None

        def capture_executor(strategy):
            nonlocal captured_strategy
            captured_strategy = strategy
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with (
            patch("workflows.shared.llm_utils.structured.executors.get_executor", side_effect=capture_executor),
            patch("workflows.shared.llm_utils.structured.retry.with_retries", return_value=mock_result),
        ):
            await invoke(
                tier=ModelTier.SONNET,
                system="Research topic.",
                user="Find info about...",
                schema=TestOutput,
                config=InvokeConfig(tools=[MagicMock()]),
            )

        assert captured_strategy == StructuredOutputStrategy.TOOL_AGENT

    @pytest.mark.asyncio
    async def test_schema_without_tools_uses_langchain_structured(self):
        """Without tools, should use LANGCHAIN_STRUCTURED strategy."""
        from pydantic import BaseModel
        from workflows.shared.llm_utils.structured.types import (
            StructuredOutputResult,
            StructuredOutputStrategy,
        )

        class TestOutput(BaseModel):
            value: str

        mock_result = StructuredOutputResult.ok(
            value=TestOutput(value="result"),
            strategy=StructuredOutputStrategy.LANGCHAIN_STRUCTURED,
        )

        captured_strategy = None

        def capture_executor(strategy):
            nonlocal captured_strategy
            captured_strategy = strategy
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=mock_result)
            return mock_executor

        with (
            patch("workflows.shared.llm_utils.structured.executors.get_executor", side_effect=capture_executor),
            patch("workflows.shared.llm_utils.structured.retry.with_retries", return_value=mock_result),
        ):
            await invoke(
                tier=ModelTier.SONNET,
                system="Extract.",
                user="Content",
                schema=TestOutput,
            )

        assert captured_strategy == StructuredOutputStrategy.LANGCHAIN_STRUCTURED


class TestInvokeBatchContextManager:
    """Tests for invoke_batch() context manager."""

    @pytest.mark.asyncio
    async def test_accumulates_requests(self):
        """Requests added via add() should be accumulated."""
        from core.llm_broker import LLMResponse

        futures = [asyncio.Future() for _ in range(3)]
        for i, f in enumerate(futures):
            f.set_result(
                LLMResponse(
                    request_id=f"test-{i}",
                    content=f"Response {i}",
                    success=True,
                )
            )

        mock_broker = MagicMock()
        mock_broker.batch_group.return_value.__aenter__ = AsyncMock()
        mock_broker.batch_group.return_value.__aexit__ = AsyncMock()
        mock_broker.request = AsyncMock(side_effect=futures)

        with patch("core.llm_broker.get_broker", return_value=mock_broker):
            async with invoke_batch() as batch:
                batch.add(tier=ModelTier.HAIKU, system="S1", user="U1")
                batch.add(tier=ModelTier.HAIKU, system="S2", user="U2")
                batch.add(tier=ModelTier.SONNET, system="S3", user="U3")

            results = await batch.results()

        assert len(results) == 3
        assert mock_broker.request.call_count == 3

    @pytest.mark.asyncio
    async def test_results_not_available_before_exit(self):
        """Calling results() before context exit should raise."""
        mock_broker = MagicMock()
        mock_broker.batch_group.return_value.__aenter__ = AsyncMock()
        mock_broker.batch_group.return_value.__aexit__ = AsyncMock()

        batch = InvokeBatch()
        batch.add(tier=ModelTier.HAIKU, system="S", user="U")

        with pytest.raises(RuntimeError, match="Results not available"):
            await batch.results()

    @pytest.mark.asyncio
    async def test_empty_batch_returns_empty_list(self):
        """Empty batch should return empty list."""
        mock_broker = MagicMock()
        mock_broker.batch_group.return_value.__aenter__ = AsyncMock()
        mock_broker.batch_group.return_value.__aexit__ = AsyncMock()
        mock_broker.request = AsyncMock()

        with patch("core.llm_broker.get_broker", return_value=mock_broker):
            async with invoke_batch() as batch:
                pass  # No adds

            results = await batch.results()

        assert results == []
