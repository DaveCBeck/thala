"""Tests for broker-routed tool agent runner."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from core.llm_broker.schemas import BatchPolicy, LLMResponse
from workflows.shared.llm_utils.structured.executors.broker_agent_runner import (
    _add_cache_breakpoint,
    _build_cache_control,
    _langchain_tool_to_anthropic,
    _schema_to_anthropic_tool,
    run_tool_agent_via_broker,
)


class SampleOutput(BaseModel):
    """Test schema for structured output."""

    summary: str
    citations: list[str]


def _make_mock_tool(name: str, description: str, schema_props: dict) -> MagicMock:
    """Create a mock LangChain BaseTool."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.args_schema = MagicMock()
    tool.args_schema.model_json_schema.return_value = {
        "type": "object",
        "properties": schema_props,
    }
    return tool


class TestLangchainToolToAnthropic:
    def test_basic_conversion(self):
        tool = _make_mock_tool("search", "Search papers", {"query": {"type": "string"}})
        result = _langchain_tool_to_anthropic(tool)
        assert result == {
            "name": "search",
            "description": "Search papers",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }

    def test_no_description(self):
        tool = _make_mock_tool("search", None, {})
        tool.description = None
        result = _langchain_tool_to_anthropic(tool)
        assert result["description"] == ""

    def test_no_args_schema(self):
        tool = MagicMock()
        tool.name = "simple_tool"
        tool.description = "A simple tool"
        tool.args_schema = None
        result = _langchain_tool_to_anthropic(tool)
        assert result["input_schema"] == {"type": "object", "properties": {}}


class TestSchemaToAnthropicTool:
    def test_basic_conversion(self):
        result = _schema_to_anthropic_tool(SampleOutput)
        assert result["name"] == "SampleOutput"
        assert "input_schema" in result
        assert "properties" in result["input_schema"]
        assert "summary" in result["input_schema"]["properties"]
        assert "citations" in result["input_schema"]["properties"]


def _make_mock_broker():
    """Create a mock broker with flush as async no-op."""
    broker = AsyncMock()
    broker.request = AsyncMock()
    broker.flush = AsyncMock()
    return broker


class TestRunToolAgentViaBroker:
    @pytest.fixture
    def mock_broker(self):
        return _make_mock_broker()

    @pytest.fixture
    def mock_tools(self):
        tool = _make_mock_tool("search", "Search papers", {"query": {"type": "string"}})
        tool.ainvoke = AsyncMock(return_value="Paper: ML basics")
        return [tool]

    @pytest.mark.asyncio
    async def test_no_tool_calls(self, mock_broker, mock_tools):
        """Model returns text only, no tool calls — goes straight to extraction."""
        import asyncio

        # First call: model returns text (no tool_use blocks)
        no_tools_response = LLMResponse(
            request_id="r1",
            content="Some text",
            success=True,
            stop_reason="end_turn",
            content_blocks=[{"type": "text", "text": "Some text"}],
        )

        # Second call: structured extraction
        extraction_response = LLMResponse(
            request_id="r2",
            content=json.dumps({"summary": "test summary", "citations": ["cite1"]}),
            success=True,
            stop_reason="end_turn",
            content_blocks=[{
                "type": "tool_use",
                "id": "toolu_ext",
                "name": "SampleOutput",
                "input": {"summary": "test summary", "citations": ["cite1"]},
            }],
        )

        # Create futures that resolve immediately
        loop = asyncio.get_event_loop()

        future1 = loop.create_future()
        future1.set_result(no_tools_response)
        future2 = loop.create_future()
        future2.set_result(extraction_response)

        mock_broker.request = AsyncMock(side_effect=[future1, future2])

        with patch(
            "workflows.shared.llm_utils.structured.executors.broker_agent_runner.get_broker",
            return_value=mock_broker,
        ):
            result = await run_tool_agent_via_broker(
                tools=mock_tools,
                system_prompt="You are helpful.",
                user_prompt="Summarize papers",
                output_schema=SampleOutput,
                tier=MagicMock(value="claude-sonnet-4-5-20250929"),
                policy=BatchPolicy.PREFER_SPEED,
                max_tokens=4096,
            )

        assert result.summary == "test summary"
        assert result.citations == ["cite1"]
        assert mock_broker.request.call_count == 2

    @pytest.mark.asyncio
    async def test_with_tool_calls(self, mock_broker, mock_tools):
        """Model makes a tool call, then extraction."""
        import asyncio

        loop = asyncio.get_event_loop()

        # Call 1: model requests tool use
        tool_call_response = LLMResponse(
            request_id="r1",
            content=json.dumps({"query": "ML basics"}),
            success=True,
            stop_reason="tool_use",
            content_blocks=[
                {"type": "text", "text": "Let me search"},
                {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": "search",
                    "input": {"query": "ML basics"},
                },
            ],
        )

        # Call 2: model returns text (done with tools)
        no_tools_response = LLMResponse(
            request_id="r2",
            content="Found papers",
            success=True,
            stop_reason="end_turn",
            content_blocks=[{"type": "text", "text": "Found papers"}],
        )

        # Call 3: structured extraction
        extraction_response = LLMResponse(
            request_id="r3",
            content=json.dumps({"summary": "ML summary", "citations": ["c1"]}),
            success=True,
            stop_reason="end_turn",
        )

        future1 = loop.create_future()
        future1.set_result(tool_call_response)
        future2 = loop.create_future()
        future2.set_result(no_tools_response)
        future3 = loop.create_future()
        future3.set_result(extraction_response)

        mock_broker.request = AsyncMock(side_effect=[future1, future2, future3])

        with patch(
            "workflows.shared.llm_utils.structured.executors.broker_agent_runner.get_broker",
            return_value=mock_broker,
        ):
            result = await run_tool_agent_via_broker(
                tools=mock_tools,
                system_prompt="You are helpful.",
                user_prompt="Summarize papers",
                output_schema=SampleOutput,
                tier=MagicMock(value="claude-sonnet-4-5-20250929"),
                policy=BatchPolicy.PREFER_SPEED,
                max_tokens=4096,
            )

        assert result.summary == "ML summary"
        assert mock_broker.request.call_count == 3
        # Verify tool was called
        mock_tools[0].ainvoke.assert_called_once_with({"query": "ML basics"})

    @pytest.mark.asyncio
    async def test_broker_failure_raises(self, mock_broker, mock_tools):
        """Broker returning success=False raises RuntimeError."""
        import asyncio

        loop = asyncio.get_event_loop()
        failure_response = LLMResponse(
            request_id="r1",
            content=None,
            success=False,
            error="Rate limited",
        )
        future = loop.create_future()
        future.set_result(failure_response)

        mock_broker.request = AsyncMock(return_value=future)

        with patch(
            "workflows.shared.llm_utils.structured.executors.broker_agent_runner.get_broker",
            return_value=mock_broker,
        ):
            with pytest.raises(RuntimeError, match="Broker tool agent request failed"):
                await run_tool_agent_via_broker(
                    tools=mock_tools,
                    system_prompt=None,
                    user_prompt="test",
                    output_schema=SampleOutput,
                    tier=MagicMock(value="claude-sonnet-4-5-20250929"),
                    policy=BatchPolicy.PREFER_SPEED,
                )

    @pytest.mark.asyncio
    async def test_max_tool_calls_limit(self, mock_broker, mock_tools):
        """Loop terminates after max_tool_calls."""
        import asyncio

        loop = asyncio.get_event_loop()

        # Always return tool_use to test the limit
        tool_response = LLMResponse(
            request_id="r",
            content=json.dumps({"query": "q"}),
            success=True,
            stop_reason="tool_use",
            content_blocks=[{
                "type": "tool_use",
                "id": "toolu_x",
                "name": "search",
                "input": {"query": "q"},
            }],
        )

        extraction_response = LLMResponse(
            request_id="re",
            content=json.dumps({"summary": "s", "citations": []}),
            success=True,
        )

        def make_future(resp):
            f = loop.create_future()
            f.set_result(resp)
            return f

        # 3 tool calls + 1 extraction = 4 broker calls
        mock_broker.request = AsyncMock(
            side_effect=[
                make_future(tool_response),
                make_future(tool_response),
                make_future(tool_response),
                make_future(extraction_response),
            ]
        )

        with patch(
            "workflows.shared.llm_utils.structured.executors.broker_agent_runner.get_broker",
            return_value=mock_broker,
        ):
            result = await run_tool_agent_via_broker(
                tools=mock_tools,
                system_prompt=None,
                user_prompt="test",
                output_schema=SampleOutput,
                tier=MagicMock(value="claude-sonnet-4-5-20250929"),
                policy=BatchPolicy.PREFER_SPEED,
                max_tool_calls=3,
            )

        assert result.summary == "s"
        # 3 tool loop calls + 1 extraction = 4
        assert mock_broker.request.call_count == 4


class TestBuildCacheControl:
    def test_default_5m(self):
        result = _build_cache_control("5m")
        assert result == {"type": "ephemeral"}

    def test_1h_ttl(self):
        result = _build_cache_control("1h")
        assert result == {"type": "ephemeral", "ttl": "1h"}


class TestAddCacheBreakpoint:
    def test_list_content(self):
        """Adds cache_control to last block in list content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result1"},
                    {"type": "tool_result", "tool_use_id": "t2", "content": "result2"},
                ],
            }
        ]
        _add_cache_breakpoint(messages, "5m")
        last_block = messages[0]["content"][-1]
        assert last_block["cache_control"] == {"type": "ephemeral"}
        # First block unchanged
        assert "cache_control" not in messages[0]["content"][0]

    def test_list_content_1h(self):
        """1h TTL is set correctly."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ]
        _add_cache_breakpoint(messages, "1h")
        assert messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_string_content(self):
        """Converts string content to list with cache_control."""
        messages = [{"role": "user", "content": "hello world"}]
        _add_cache_breakpoint(messages, "5m")
        assert messages[0]["content"] == [
            {"type": "text", "text": "hello world", "cache_control": {"type": "ephemeral"}}
        ]

    def test_empty_messages(self):
        """No-op on empty messages list."""
        messages: list[dict] = []
        _add_cache_breakpoint(messages, "5m")
        assert messages == []

    def test_preserves_existing_fields(self):
        """Original block fields are preserved when adding cache_control."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "data"},
                ],
            }
        ]
        _add_cache_breakpoint(messages, "5m")
        block = messages[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "t1"
        assert block["content"] == "data"
        assert block["cache_control"] == {"type": "ephemeral"}


class TestCachingInAgentLoop:
    """Tests that cache breakpoints are applied correctly during the agent loop."""

    @pytest.fixture
    def mock_broker(self):
        return _make_mock_broker()

    @pytest.fixture
    def mock_tools(self):
        tool = _make_mock_tool("search", "Search papers", {"query": {"type": "string"}})
        tool.ainvoke = AsyncMock(return_value="Paper: ML basics")
        return [tool]

    @pytest.mark.asyncio
    async def test_tool_defs_get_cache_control(self, mock_broker, mock_tools):
        """Tool definitions get cache_control when cache_ttl is set."""
        import asyncio

        loop = asyncio.get_event_loop()

        # Return text only (no tool calls) then extraction
        no_tools = LLMResponse(
            request_id="r1", content="text", success=True,
            content_blocks=[{"type": "text", "text": "text"}],
        )
        extraction = LLMResponse(
            request_id="r2",
            content=json.dumps({"summary": "s", "citations": []}),
            success=True,
        )

        f1 = loop.create_future()
        f1.set_result(no_tools)
        f2 = loop.create_future()
        f2.set_result(extraction)
        mock_broker.request = AsyncMock(side_effect=[f1, f2])

        with patch(
            "workflows.shared.llm_utils.structured.executors.broker_agent_runner.get_broker",
            return_value=mock_broker,
        ):
            await run_tool_agent_via_broker(
                tools=mock_tools,
                system_prompt="sys",
                user_prompt="test",
                output_schema=SampleOutput,
                tier=MagicMock(value="claude-sonnet-4-5-20250929"),
                policy=BatchPolicy.PREFER_SPEED,
                cache_ttl="1h",
            )

        # First call should have tool defs with cache_control on last tool
        first_call = mock_broker.request.call_args_list[0]
        tools_arg = first_call.kwargs["tools"]
        assert tools_arg[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    @pytest.mark.asyncio
    async def test_no_cache_when_ttl_none(self, mock_broker, mock_tools):
        """No cache_control added when cache_ttl is None."""
        import asyncio

        loop = asyncio.get_event_loop()

        no_tools = LLMResponse(
            request_id="r1", content="text", success=True,
            content_blocks=[{"type": "text", "text": "text"}],
        )
        extraction = LLMResponse(
            request_id="r2",
            content=json.dumps({"summary": "s", "citations": []}),
            success=True,
        )

        f1 = loop.create_future()
        f1.set_result(no_tools)
        f2 = loop.create_future()
        f2.set_result(extraction)
        mock_broker.request = AsyncMock(side_effect=[f1, f2])

        with patch(
            "workflows.shared.llm_utils.structured.executors.broker_agent_runner.get_broker",
            return_value=mock_broker,
        ):
            await run_tool_agent_via_broker(
                tools=mock_tools,
                system_prompt="sys",
                user_prompt="test",
                output_schema=SampleOutput,
                tier=MagicMock(value="claude-sonnet-4-5-20250929"),
                policy=BatchPolicy.PREFER_SPEED,
                cache_ttl=None,
            )

        # Tools should NOT have cache_control
        first_call = mock_broker.request.call_args_list[0]
        tools_arg = first_call.kwargs["tools"]
        assert "cache_control" not in tools_arg[-1]

    @pytest.mark.asyncio
    async def test_message_cache_on_round_2(self, mock_broker, mock_tools):
        """Cache breakpoint is added to messages on round 2+ but not round 1."""
        import asyncio

        loop = asyncio.get_event_loop()

        # Round 1: model calls a tool
        tool_call = LLMResponse(
            request_id="r1", content="", success=True, stop_reason="tool_use",
            content_blocks=[{
                "type": "tool_use", "id": "t1", "name": "search", "input": {"query": "q"},
            }],
        )
        # Round 2: model returns text (done)
        text_resp = LLMResponse(
            request_id="r2", content="done", success=True,
            content_blocks=[{"type": "text", "text": "done"}],
        )
        # Extraction
        extraction = LLMResponse(
            request_id="r3",
            content=json.dumps({"summary": "s", "citations": []}),
            success=True,
        )

        futures = []
        for resp in [tool_call, text_resp, extraction]:
            f = loop.create_future()
            f.set_result(resp)
            futures.append(f)
        mock_broker.request = AsyncMock(side_effect=futures)

        with patch(
            "workflows.shared.llm_utils.structured.executors.broker_agent_runner.get_broker",
            return_value=mock_broker,
        ):
            await run_tool_agent_via_broker(
                tools=mock_tools,
                system_prompt="sys",
                user_prompt="test",
                output_schema=SampleOutput,
                tier=MagicMock(value="claude-sonnet-4-5-20250929"),
                policy=BatchPolicy.PREFER_SPEED,
                cache_ttl="1h",
            )

        # Round 1 (call_args_list[0]): no message cache breakpoint
        round1_messages = mock_broker.request.call_args_list[0].kwargs["messages"]
        first_msg_content = round1_messages[0]["content"]
        if isinstance(first_msg_content, str):
            pass  # string content = no cache_control, correct
        else:
            assert "cache_control" not in first_msg_content[-1]

        # Round 2 (call_args_list[1]): should have cache breakpoint on last message
        round2_messages = mock_broker.request.call_args_list[1].kwargs["messages"]
        last_msg = round2_messages[-1]
        last_content = last_msg["content"]
        if isinstance(last_content, list):
            assert "cache_control" in last_content[-1]
            assert last_content[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    @pytest.mark.asyncio
    async def test_final_extraction_has_cache(self, mock_broker, mock_tools):
        """Final extraction call has cache breakpoints on tools and messages."""
        import asyncio

        loop = asyncio.get_event_loop()

        # No tool calls, straight to extraction
        no_tools = LLMResponse(
            request_id="r1", content="text", success=True,
            content_blocks=[{"type": "text", "text": "text"}],
        )
        extraction = LLMResponse(
            request_id="r2",
            content=json.dumps({"summary": "s", "citations": []}),
            success=True,
        )

        f1 = loop.create_future()
        f1.set_result(no_tools)
        f2 = loop.create_future()
        f2.set_result(extraction)
        mock_broker.request = AsyncMock(side_effect=[f1, f2])

        with patch(
            "workflows.shared.llm_utils.structured.executors.broker_agent_runner.get_broker",
            return_value=mock_broker,
        ):
            await run_tool_agent_via_broker(
                tools=mock_tools,
                system_prompt="sys",
                user_prompt="test",
                output_schema=SampleOutput,
                tier=MagicMock(value="claude-sonnet-4-5-20250929"),
                policy=BatchPolicy.PREFER_SPEED,
                cache_ttl="5m",
            )

        # Final extraction call (call_args_list[1])
        final_call = mock_broker.request.call_args_list[1]
        # Schema tool should have cache_control
        final_tools = final_call.kwargs["tools"]
        assert final_tools[-1]["cache_control"] == {"type": "ephemeral"}
        # Messages should have cache breakpoint
        final_messages = final_call.kwargs["messages"]
        last_msg = final_messages[-1]
        content = last_msg["content"]
        if isinstance(content, list):
            assert "cache_control" in content[-1]
