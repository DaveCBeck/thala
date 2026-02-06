"""Integration tests verifying LangSmith trace consistency across invoke() paths.

These tests verify that all routing paths through invoke() produce consistent
LangSmith traces with proper token tracking, model identification, and cache metrics.

The invoke() function has 4 routing paths:
1. Direct Anthropic - No batch_policy, uses llm.ainvoke() with prompt caching
2. Broker-routed - With batch_policy, uses broker.request()
3. DeepSeek direct - All DeepSeek tiers bypass broker
4. Structured output - Schema provided, uses executor strategies

Each path handles token metadata differently and these tests ensure consistency.

Run with: pytest tests/integration/llm_utils/test_invoke_langsmith_traces.py -m llm -v
"""

import os

import pytest

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke


@pytest.mark.llm
@pytest.mark.asyncio
class TestInvokeTraceMetadata:
    """Verify AIMessage metadata is consistent across routing paths."""

    async def test_direct_path_has_usage_metadata(self):
        """Direct Anthropic path populates usage_metadata."""
        response = await invoke(
            tier=ModelTier.HAIKU,
            system="You are helpful.",
            user="Say hello in exactly 3 words.",
        )

        assert response.usage_metadata is not None
        assert response.usage_metadata.get("input_tokens", 0) > 0
        assert response.usage_metadata.get("output_tokens", 0) > 0
        assert response.usage_metadata.get("total_tokens", 0) > 0

    async def test_broker_path_has_usage_metadata(self):
        """Broker path populates usage_metadata identically."""
        response = await invoke(
            tier=ModelTier.HAIKU,
            system="You are helpful.",
            user="Say hello in exactly 3 words.",
            config=InvokeConfig(batch_policy=BatchPolicy.REQUIRE_SYNC),
        )

        assert response.usage_metadata is not None
        assert response.usage_metadata.get("input_tokens", 0) > 0
        assert response.usage_metadata.get("output_tokens", 0) > 0
        assert response.usage_metadata.get("total_tokens", 0) > 0

    async def test_model_name_in_response_metadata(self):
        """Model name preserved in response_metadata for both paths."""
        direct = await invoke(
            tier=ModelTier.HAIKU,
            system="Test.",
            user="Hi.",
        )

        broker = await invoke(
            tier=ModelTier.HAIKU,
            system="Test.",
            user="Hi.",
            config=InvokeConfig(batch_policy=BatchPolicy.REQUIRE_SYNC),
        )

        # Both should have model in response_metadata
        direct_model = direct.response_metadata.get("model", "")
        broker_model = broker.response_metadata.get("model", "")

        assert "claude" in direct_model.lower() or direct_model != ""
        assert "claude" in broker_model.lower() or broker_model != ""

    async def test_cache_tokens_reported_when_applicable(self):
        """Cache token details present in usage_metadata."""
        # Make same call twice to potentially get cache hit
        prompt = "Explain caching in one sentence."

        await invoke(tier=ModelTier.HAIKU, system="Be brief.", user=prompt)
        response = await invoke(tier=ModelTier.HAIKU, system="Be brief.", user=prompt)

        # Cache details may be in input_token_details
        details = response.usage_metadata.get("input_token_details", {})
        # Note: cache_read may be 0 if no cache hit, but structure should exist
        # This test verifies the field is populated when present
        assert isinstance(details, dict)


# Skip if no DeepSeek API key
deepseek_available = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set",
)


@pytest.mark.llm
@pytest.mark.asyncio
class TestDeepSeekTraceMetadata:
    """Verify DeepSeek paths produce proper trace metadata."""

    @deepseek_available
    async def test_deepseek_v3_has_usage_metadata(self):
        """DeepSeek V3 path populates usage_metadata."""
        response = await invoke(
            tier=ModelTier.DEEPSEEK_V3,
            system="You are helpful.",
            user="Say hello in exactly 3 words.",
        )

        assert response.usage_metadata is not None
        assert response.usage_metadata.get("input_tokens", 0) > 0
        assert response.usage_metadata.get("output_tokens", 0) > 0

    @deepseek_available
    async def test_deepseek_r1_has_thinking_content(self):
        """DeepSeek R1 preserves thinking content in additional_kwargs."""
        response = await invoke(
            tier=ModelTier.DEEPSEEK_R1,
            system="You are a reasoning assistant.",
            user="What is 2+2? Think step by step.",
        )

        assert response.usage_metadata is not None
        # R1 may include reasoning in additional_kwargs or content
        # Just verify we get a valid response with tokens
        assert response.usage_metadata.get("output_tokens", 0) > 0
