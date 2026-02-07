"""Integration tests verifying invoke_with_cache to invoke() migration.

These tests validate that the migrated functions work correctly with real LLM calls.
They test the specific patterns used in:
- drafting.py (Anthropic with caching, max_tokens)
- summary_agent.py (DeepSeek R1)

Run with: pytest tests/integration/llm_utils/test_invoke_migration.py -m llm -v
"""

import pytest

from workflows.shared.llm_utils import ModelTier, invoke, InvokeConfig


@pytest.mark.llm
@pytest.mark.asyncio
class TestInvokeMigrationPatterns:
    """Integration tests for patterns used in migrated files."""

    async def test_anthropic_with_cache_and_max_tokens(self):
        """Test pattern from drafting.py: Sonnet with cache and max_tokens.

        Migrated from:
            llm = get_llm(tier=ModelTier.SONNET, max_tokens=4096)
            response = await invoke_with_cache(llm, system_prompt=..., user_prompt=...)

        To:
            response = await invoke(
                tier=ModelTier.SONNET,
                system=...,
                user=...,
                config=InvokeConfig(max_tokens=4096),
            )
        """
        response = await invoke(
            tier=ModelTier.SONNET,
            system="You are a technical writer. Be concise.",
            user="Write one sentence about machine learning.",
            config=InvokeConfig(max_tokens=100),
        )

        assert response is not None
        assert hasattr(response, "content")
        assert isinstance(response.content, str)
        assert len(response.content) > 10
        # Verify we got a coherent response about ML
        assert any(
            term in response.content.lower()
            for term in ["machine", "learn", "ai", "algorithm", "data"]
        )

    async def test_deepseek_r1_simple_call(self):
        """Test pattern from summary_agent.py: DeepSeek R1 without cache config.

        Migrated from:
            llm = get_llm(tier=ModelTier.DEEPSEEK_R1)
            response = await invoke_with_cache(llm, system_prompt=..., user_prompt=...)

        To:
            response = await invoke(
                tier=ModelTier.DEEPSEEK_R1,
                system=...,
                user=...,
            )

        Note: DeepSeek routes directly (not through broker) and has automatic
        prefix-based caching.
        """
        response = await invoke(
            tier=ModelTier.DEEPSEEK_R1,
            system="You are a helpful assistant. Be very concise.",
            user="What is 2 + 2? Reply with just the number.",
        )

        assert response is not None
        assert hasattr(response, "content")
        assert isinstance(response.content, str)
        # DeepSeek R1 may include thinking, so content might be structured
        # Check for the answer somewhere in the content
        assert "4" in response.content or "four" in response.content.lower()
