"""Tests for broker-routed structured output path."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from core.llm_broker.schemas import BatchPolicy, LLMResponse


class SampleSchema(BaseModel):
    """Test schema."""

    name: str
    tags: list[str]


class TestInvokeStructuredViaBroker:
    """Tests for _invoke_structured_via_broker helper."""

    @pytest.fixture
    def mock_broker(self):
        broker = AsyncMock()
        broker.batch_group = MagicMock()
        # Make batch_group work as async context manager
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=None)
        ctx.__aexit__ = AsyncMock(return_value=None)
        broker.batch_group.return_value = ctx
        broker.request = AsyncMock()
        return broker

    @pytest.mark.asyncio
    async def test_routes_through_broker(self, mock_broker):
        """Structured output goes through broker with tool_use trick."""
        import asyncio

        loop = asyncio.get_event_loop()
        response = LLMResponse(
            request_id="test",
            content=json.dumps({"name": "test", "tags": ["a", "b"]}),
            success=True,
        )
        future = loop.create_future()
        future.set_result(response)
        mock_broker.request = AsyncMock(return_value=future)

        with (
            patch("core.llm_broker.get_broker", return_value=mock_broker),
            patch("core.llm_broker.is_broker_enabled", return_value=True),
        ):
            # Import after patching
            from workflows.shared.llm_utils.invoke import _invoke_structured_via_broker
            from workflows.shared.llm_utils.config import InvokeConfig

            result = await _invoke_structured_via_broker(
                tier=MagicMock(value="claude-sonnet-4-5-20250929"),
                system="test system",
                user_prompt="test prompt",
                config=InvokeConfig(batch_policy=BatchPolicy.PREFER_SPEED),
                schema=SampleSchema,
            )

        assert result.name == "test"
        assert result.tags == ["a", "b"]

        # Verify broker was called with tool definition
        call_kwargs = mock_broker.request.call_args.kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "SampleSchema"
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "SampleSchema"}

    @pytest.mark.asyncio
    async def test_raises_on_broker_failure(self, mock_broker):
        """Raises RuntimeError when broker returns failure."""
        import asyncio

        loop = asyncio.get_event_loop()
        response = LLMResponse(
            request_id="test",
            content=None,
            success=False,
            error="Something went wrong",
        )
        future = loop.create_future()
        future.set_result(response)
        mock_broker.request = AsyncMock(return_value=future)

        with (
            patch("core.llm_broker.get_broker", return_value=mock_broker),
            patch("core.llm_broker.is_broker_enabled", return_value=True),
        ):
            from workflows.shared.llm_utils.invoke import _invoke_structured_via_broker
            from workflows.shared.llm_utils.config import InvokeConfig

            with pytest.raises(RuntimeError, match="Broker structured request failed"):
                await _invoke_structured_via_broker(
                    tier=MagicMock(value="claude-sonnet-4-5-20250929"),
                    system="test",
                    user_prompt="test",
                    config=InvokeConfig(batch_policy=BatchPolicy.PREFER_SPEED),
                    schema=SampleSchema,
                )

    @pytest.mark.asyncio
    async def test_coerces_string_to_list(self, mock_broker):
        """Applies coerce_to_schema for LLM quirks (string -> list)."""
        import asyncio

        loop = asyncio.get_event_loop()
        # LLM returns string instead of list for tags
        response = LLMResponse(
            request_id="test",
            content=json.dumps({"name": "test", "tags": "single_tag"}),
            success=True,
        )
        future = loop.create_future()
        future.set_result(response)
        mock_broker.request = AsyncMock(return_value=future)

        with (
            patch("core.llm_broker.get_broker", return_value=mock_broker),
            patch("core.llm_broker.is_broker_enabled", return_value=True),
        ):
            from workflows.shared.llm_utils.invoke import _invoke_structured_via_broker
            from workflows.shared.llm_utils.config import InvokeConfig

            result = await _invoke_structured_via_broker(
                tier=MagicMock(value="claude-sonnet-4-5-20250929"),
                system="test",
                user_prompt="test",
                config=InvokeConfig(batch_policy=BatchPolicy.PREFER_SPEED),
                schema=SampleSchema,
            )

        # coerce_to_schema should convert string to single-item list
        assert result.tags == ["single_tag"]
