"""Tests for extended message_params functionality (messages, content_blocks)."""

import json
from unittest.mock import MagicMock

from core.llm_broker.message_params import (
    build_message_params,
    parse_response_content_with_blocks,
)
from core.llm_broker.schemas import LLMRequest, LLMResponse


class TestBuildMessageParamsWithMessages:
    """Tests for build_message_params with messages field."""

    def test_uses_prompt_when_messages_none(self):
        """Default behavior: wraps prompt as single user message."""
        request = LLMRequest.create(prompt="Hello", model="claude-sonnet-4-5-20250929")
        params = build_message_params(request)
        assert params["messages"] == [{"role": "user", "content": "Hello"}]

    def test_uses_messages_when_provided(self):
        """When messages is set, uses it directly instead of wrapping prompt."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "user", "content": "How are you?"},
        ]
        request = LLMRequest.create(
            prompt="",
            model="claude-sonnet-4-5-20250929",
            messages=messages,
        )
        params = build_message_params(request)
        assert params["messages"] == messages

    def test_messages_with_tool_results(self):
        """Messages containing tool_result blocks are passed through."""
        messages = [
            {"role": "user", "content": "Search for papers"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_123", "name": "search", "input": {"q": "test"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "result"},
                ],
            },
        ]
        request = LLMRequest.create(
            prompt="",
            model="claude-sonnet-4-5-20250929",
            messages=messages,
        )
        params = build_message_params(request)
        assert params["messages"] == messages
        assert len(params["messages"]) == 3


    def test_adaptive_thinking_params(self):
        """When effort is set, builds adaptive thinking + output_config params."""
        request = LLMRequest.create(
            prompt="Think hard",
            model="claude-opus-4-6",
            effort="high",
        )
        params = build_message_params(request)
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "high"}
        # No max_tokens bump — adaptive thinking doesn't use explicit budget
        assert params["max_tokens"] == 4096

    def test_no_thinking_params_without_effort(self):
        """Without effort, no thinking or output_config params."""
        request = LLMRequest.create(prompt="Hello", model="claude-sonnet-4-5-20250929")
        params = build_message_params(request)
        assert "thinking" not in params
        assert "output_config" not in params


class TestParseResponseContentWithBlocks:
    """Tests for parse_response_content_with_blocks."""

    def _make_block(self, block_type, **kwargs):
        """Create a mock content block."""
        block = MagicMock()
        block.type = block_type
        for k, v in kwargs.items():
            setattr(block, k, v)
        return block

    def _make_response(self, blocks):
        """Create a mock Anthropic response."""
        response = MagicMock()
        response.content = blocks
        return response

    def test_text_only(self):
        response = self._make_response([
            self._make_block("text", text="Hello world"),
        ])
        content, thinking, blocks = parse_response_content_with_blocks(response)
        assert content == "Hello world"
        assert thinking is None
        assert blocks == [{"type": "text", "text": "Hello world"}]

    def test_thinking_and_text(self):
        response = self._make_response([
            self._make_block("thinking", thinking="Let me think..."),
            self._make_block("text", text="The answer"),
        ])
        content, thinking, blocks = parse_response_content_with_blocks(response)
        assert content == "The answer"
        assert thinking == "Let me think..."
        assert len(blocks) == 2
        assert blocks[0] == {"type": "thinking", "thinking": "Let me think..."}
        assert blocks[1] == {"type": "text", "text": "The answer"}

    def test_tool_use_preserves_id(self):
        """tool_use blocks preserve id and name in content_blocks."""
        response = self._make_response([
            self._make_block(
                "tool_use",
                id="toolu_abc123",
                name="search_papers",
                input={"query": "machine learning"},
            ),
        ])
        content, thinking, blocks = parse_response_content_with_blocks(response)
        assert content == json.dumps({"query": "machine learning"})
        assert blocks == [{
            "type": "tool_use",
            "id": "toolu_abc123",
            "name": "search_papers",
            "input": {"query": "machine learning"},
        }]

    def test_tool_use_with_output_wrapper(self):
        """$output wrapper is unwrapped in content but original kept in blocks."""
        response = self._make_response([
            self._make_block(
                "tool_use",
                id="toolu_xyz",
                name="output_tool",
                input={"$output": {"field": "value"}},
            ),
        ])
        content, thinking, blocks = parse_response_content_with_blocks(response)
        # content should be unwrapped
        assert json.loads(content) == {"field": "value"}
        # blocks should have original input for API compatibility
        assert blocks[0]["input"] == {"$output": {"field": "value"}}

    def test_mixed_text_and_tool_use(self):
        """Response with both text and tool_use blocks."""
        response = self._make_response([
            self._make_block("text", text="I'll search for that"),
            self._make_block(
                "tool_use",
                id="toolu_001",
                name="search",
                input={"q": "test"},
            ),
        ])
        content, thinking, blocks = parse_response_content_with_blocks(response)
        # content is the last block processed (tool_use input)
        assert content == json.dumps({"q": "test"})
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["id"] == "toolu_001"


class TestLLMRequestMessagesSerialization:
    """Tests for LLMRequest messages field serialization."""

    def test_to_dict_includes_messages(self):
        messages = [{"role": "user", "content": "hi"}]
        request = LLMRequest.create(
            prompt="",
            model="claude-sonnet-4-5-20250929",
            messages=messages,
        )
        d = request.to_dict()
        assert d["messages"] == messages

    def test_to_dict_messages_none(self):
        request = LLMRequest.create(prompt="hello", model="claude-sonnet-4-5-20250929")
        d = request.to_dict()
        assert d["messages"] is None

    def test_from_dict_roundtrip(self):
        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
        request = LLMRequest.create(
            prompt="",
            model="claude-sonnet-4-5-20250929",
            messages=messages,
        )
        d = request.to_dict()
        restored = LLMRequest.from_dict(d)
        assert restored.messages == messages

    def test_from_dict_no_messages_field(self):
        """Old serialized data without messages field should work."""
        d = {
            "request_id": "test-id",
            "prompt": "hello",
            "model": "claude-sonnet-4-5-20250929",
            "policy": "prefer_speed",
            "state": "queued",
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        request = LLMRequest.from_dict(d)
        assert request.messages is None


class TestLLMResponseContentBlocks:
    """Tests for LLMResponse content_blocks field."""

    def test_to_dict_includes_content_blocks(self):
        blocks = [{"type": "text", "text": "hello"}]
        response = LLMResponse(
            request_id="test",
            content="hello",
            content_blocks=blocks,
        )
        d = response.to_dict()
        assert d["content_blocks"] == blocks

    def test_to_dict_content_blocks_none(self):
        response = LLMResponse(request_id="test", content="hello")
        d = response.to_dict()
        assert d["content_blocks"] is None
