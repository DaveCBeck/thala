"""Integration tests for invoke() that make real LLM API calls.

These tests verify the full flow: call → LLM API → response parsing.
They require API keys and cost money, so they're marked with @pytest.mark.llm.

Run with: pytest tests/integration/llm_utils/test_invoke_llm.py -m llm -v
Skip with: pytest -m "not llm"
"""

import pytest

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import ModelTier, invoke, invoke_batch, InvokeConfig


@pytest.mark.llm
@pytest.mark.asyncio
class TestInvokeLLM:
    """Integration tests for invoke() with real LLM calls."""

    async def test_invoke_simple_haiku_call(self):
        """Verify invoke() can make a simple Haiku call and get a response."""
        response = await invoke(
            tier=ModelTier.HAIKU,
            system="You are a helpful assistant. Respond with exactly one word.",
            user="What is 2+2? Reply with just the number.",
        )

        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0
        # Should contain "4" or "four" somewhere in the response
        assert "4" in response.content or "four" in response.content.lower()

    async def test_invoke_with_batch_policy(self):
        """Verify invoke() works with batch_policy set (routes through broker)."""
        response = await invoke(
            tier=ModelTier.HAIKU,
            system="You are a helpful assistant.",
            user="Say 'hello' and nothing else.",
            config=InvokeConfig(
                batch_policy=BatchPolicy.PREFER_BALANCE,
                max_tokens=50,
            ),
        )

        assert response is not None
        assert hasattr(response, "content")
        assert "hello" in response.content.lower()

    async def test_invoke_batch_input(self):
        """Verify invoke() handles list input (batch) correctly."""
        responses = await invoke(
            tier=ModelTier.HAIKU,
            system="You are a helpful assistant. Reply with just YES or NO.",
            user=[
                "Is the sky blue?",
                "Is water wet?",
            ],
        )

        assert isinstance(responses, list)
        assert len(responses) == 2
        for response in responses:
            assert hasattr(response, "content")
            assert len(response.content) > 0


@pytest.mark.llm
@pytest.mark.asyncio
class TestInvokeBatchLLM:
    """Integration tests for invoke_batch() with real LLM calls."""

    async def test_invoke_batch_context_manager(self):
        """Verify invoke_batch() context manager works with real LLM calls."""
        async with invoke_batch() as batch:
            batch.add(
                tier=ModelTier.HAIKU,
                system="Reply with just the number.",
                user="What is 1+1?",
            )
            batch.add(
                tier=ModelTier.HAIKU,
                system="Reply with just the number.",
                user="What is 2+2?",
            )

        results = await batch.results()

        assert len(results) == 2
        # Check for numeric or word form
        assert "2" in results[0].content or "two" in results[0].content.lower()
        assert "4" in results[1].content or "four" in results[1].content.lower()


@pytest.mark.llm
@pytest.mark.asyncio
class TestQueryTranslatorLLM:
    """Integration tests for query_translator with real LLM calls."""

    async def test_translate_query_to_spanish(self):
        """Verify translate_query works end-to-end."""
        from workflows.shared.language.query_translator import translate_query, clear_query_cache

        # Clear cache to ensure we hit the LLM
        clear_query_cache()

        result = await translate_query(
            query="machine learning",
            target_language_code="es",
            target_language_name="Spanish",
        )

        assert result is not None
        assert len(result) > 0
        # Should be Spanish, not English
        assert result.lower() != "machine learning"
        # Common Spanish translations
        assert any(
            term in result.lower()
            for term in ["aprendizaje", "máquina", "automático"]
        )

    async def test_translate_query_english_passthrough(self):
        """Verify English queries are passed through without LLM call."""
        from workflows.shared.language.query_translator import translate_query

        result = await translate_query(
            query="machine learning",
            target_language_code="en",
            target_language_name="English",
        )

        assert result == "machine learning"

    async def test_translate_queries_batch(self):
        """Verify translate_queries batch translation works."""
        from workflows.shared.language.query_translator import translate_queries, clear_query_cache

        # Clear cache to ensure we hit the LLM
        clear_query_cache()

        results = await translate_queries(
            queries=["hello", "goodbye"],
            target_language_code="fr",
            target_language_name="French",
        )

        assert len(results) == 2
        # Should be French translations
        assert results[0].lower() != "hello"
        assert results[1].lower() != "goodbye"
        # Common French translations
        assert any(term in results[0].lower() for term in ["bonjour", "salut"])
        assert any(term in results[1].lower() for term in ["au revoir", "adieu", "bye"])
