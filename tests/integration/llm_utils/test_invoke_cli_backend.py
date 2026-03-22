"""Integration tests for the CLI backend (claude -p) that make real calls.

These tests verify the full flow: invoke() → claude -p subprocess → response parsing.
They require a Claude Code subscription and hit real LLM calls.

Run with: pytest tests/integration/llm_utils/test_invoke_cli_backend.py -m llm -v
Skip with: pytest -m "not llm"

Prerequisites:
  - `claude` CLI installed and authenticated
  - THALA_LLM_BACKEND=cli in .env (or set in environment)
"""

import pytest
from pydantic import BaseModel, Field

from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke
from workflows.shared.llm_utils.cli_backend import (
    invoke_structured_via_cli,
    invoke_via_cli,
)


# -- Test schemas --


class MathAnswer(BaseModel):
    """Simple structured output for math questions."""

    answer: int = Field(description="The numeric answer")
    explanation: str = Field(description="Brief explanation of the calculation")


class SentimentResult(BaseModel):
    """Structured output for sentiment analysis."""

    sentiment: str = Field(description="One of: positive, negative, neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")


class PaperMetadata(BaseModel):
    """More complex structured output mimicking real usage."""

    title: str = Field(description="Paper title")
    themes: list[str] = Field(description="Key themes (2-4 items)")
    methodology: str = Field(description="Brief methodology description")
    relevance_score: float = Field(description="Relevance score 0-1")


# -- Direct CLI backend calls --


@pytest.mark.llm
@pytest.mark.asyncio
class TestCliBackendTextDirect:
    """Test invoke_via_cli() directly with real claude -p calls."""

    async def test_haiku_text(self):
        """Haiku text call via CLI backend."""
        result = await invoke_via_cli(
            ModelTier.HAIKU,
            "You are a helpful assistant. Respond with exactly one word.",
            "What is 2+2? Reply with just the number.",
        )
        assert result.content is not None
        assert "4" in result.content or "four" in result.content.lower()

    async def test_sonnet_text(self):
        """Sonnet text call via CLI backend."""
        result = await invoke_via_cli(
            ModelTier.SONNET,
            "You are a helpful assistant. Be very brief.",
            "Name one planet in our solar system. One word only.",
        )
        assert result.content is not None
        assert len(result.content.strip()) > 0

    async def test_opus_text(self):
        """Opus text call via CLI backend."""
        result = await invoke_via_cli(
            ModelTier.OPUS,
            "You are a helpful assistant. Be very brief.",
            "What colour is the sky on a clear day? One word.",
        )
        assert result.content is not None
        assert len(result.content.strip()) > 0

    async def test_text_with_effort(self):
        """Text call with --effort flag."""
        result = await invoke_via_cli(
            ModelTier.SONNET,
            "You are a helpful assistant.",
            "Explain why 1+1=2 in one sentence.",
            effort="low",
        )
        assert result.content is not None
        assert len(result.content.strip()) > 0


@pytest.mark.llm
@pytest.mark.asyncio
class TestCliBackendStructuredDirect:
    """Test invoke_structured_via_cli() directly with real claude -p calls."""

    async def test_simple_structured(self):
        """Structured output with a simple schema."""
        result = await invoke_structured_via_cli(
            ModelTier.HAIKU,
            "You are a math assistant.",
            "What is 7 * 8?",
            MathAnswer,
        )
        assert isinstance(result, MathAnswer)
        assert result.answer == 56
        assert len(result.explanation) > 0

    async def test_enum_like_structured(self):
        """Structured output with constrained string field."""
        result = await invoke_structured_via_cli(
            ModelTier.HAIKU,
            "You are a sentiment analyzer.",
            "Analyze the sentiment: 'I love sunny days!'",
            SentimentResult,
        )
        assert isinstance(result, SentimentResult)
        assert result.sentiment in ("positive", "negative", "neutral")
        assert 0 <= result.confidence <= 1

    async def test_complex_structured(self):
        """Structured output with nested types (list, float)."""
        result = await invoke_structured_via_cli(
            ModelTier.SONNET,
            "You are an academic paper analyst.",
            "Analyze this paper: 'Deep Learning for Natural Language Processing: "
            "A Survey'. It covers transformer architectures, attention mechanisms, "
            "and their applications in NLP tasks like translation and summarization.",
            PaperMetadata,
        )
        assert isinstance(result, PaperMetadata)
        assert len(result.title) > 0
        assert len(result.themes) >= 2
        assert len(result.methodology) > 0
        assert 0 <= result.relevance_score <= 1

    async def test_structured_with_effort(self):
        """Structured output with --effort flag."""
        result = await invoke_structured_via_cli(
            ModelTier.SONNET,
            "You are a math assistant.",
            "What is 12 * 12?",
            MathAnswer,
            effort="low",
        )
        assert isinstance(result, MathAnswer)
        assert result.answer == 144


# -- Full invoke() round-trip via CLI backend --


@pytest.mark.llm
@pytest.mark.asyncio
class TestInvokeCliRoundTrip:
    """Test invoke() end-to-end with THALA_LLM_BACKEND=cli.

    These tests set the env var explicitly to ensure CLI routing,
    regardless of what's in .env.
    """

    @pytest.fixture(autouse=True)
    def _force_cli_backend(self, monkeypatch):
        monkeypatch.setenv("THALA_LLM_BACKEND", "cli")

    async def test_invoke_text_via_cli(self):
        """invoke() text call routes through CLI backend."""
        result = await invoke(
            tier=ModelTier.HAIKU,
            system="Reply with just the number.",
            user="What is 3+3?",
        )
        assert "6" in result.content

    async def test_invoke_structured_via_cli(self):
        """invoke() structured call routes through CLI backend."""
        result = await invoke(
            tier=ModelTier.HAIKU,
            system="You are a math assistant.",
            user="What is 9 * 9?",
            schema=MathAnswer,
        )
        assert isinstance(result, MathAnswer)
        assert result.answer == 81

    async def test_invoke_with_effort_via_cli(self):
        """invoke() with effort= routes through CLI backend."""
        result = await invoke(
            tier=ModelTier.SONNET,
            system="You are a sentiment analyzer.",
            user="Analyze: 'This is terrible.'",
            schema=SentimentResult,
            config=InvokeConfig(effort="low"),
        )
        assert isinstance(result, SentimentResult)
        assert result.sentiment == "negative"

    async def test_invoke_batch_text_via_cli(self):
        """invoke() with batch input unwraps to sequential CLI calls."""
        results = await invoke(
            tier=ModelTier.HAIKU,
            system="Reply with just YES or NO.",
            user=["Is the sky blue?", "Is fire cold?"],
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert "yes" in results[0].content.lower()
        assert "no" in results[1].content.lower()

    async def test_invoke_batch_structured_via_cli(self):
        """invoke() with batch structured input unwraps to sequential CLI calls."""
        results = await invoke(
            tier=ModelTier.HAIKU,
            system="You are a math assistant.",
            user=["What is 2+2?", "What is 5*5?"],
            schema=MathAnswer,
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].answer == 4
        assert results[1].answer == 25

    async def test_invoke_batch_policy_ignored_via_cli(self):
        """invoke() with batch_policy set still routes through CLI (policy ignored)."""
        from core.llm_broker import BatchPolicy

        result = await invoke(
            tier=ModelTier.HAIKU,
            system="Reply with just the number.",
            user="What is 1+1?",
            config=InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE),
        )
        assert "2" in result.content

    async def test_invoke_sonnet_1m_via_cli(self):
        """SONNET_1M tier routes through CLI as 'sonnet'."""
        result = await invoke(
            tier=ModelTier.SONNET_1M,
            system="Reply with just the number.",
            user="What is 10+10?",
        )
        assert "20" in result.content
