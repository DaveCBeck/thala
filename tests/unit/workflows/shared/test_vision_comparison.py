"""Tests for shared vision pair comparison utility."""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from workflows.shared.vision_comparison import vision_pair_select, _compare_pair


@pytest.fixture
def fake_candidates():
    """Three fake PNG byte sequences."""
    return [b"png_data_0", b"png_data_1", b"png_data_2"]


class TestVisionPairSelect:
    """Test tournament-style pair selection."""

    @pytest.mark.asyncio
    async def test_single_candidate_returns_zero(self):
        result = await vision_pair_select([b"only_one"], "criteria")
        assert result == 0

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_zero(self):
        result = await vision_pair_select([], "criteria")
        assert result == 0

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison._compare_pair")
    async def test_two_candidates_a_wins(self, mock_compare):
        mock_compare.return_value = "A"
        result = await vision_pair_select([b"img_a", b"img_b"], "criteria")
        assert result == 0
        mock_compare.assert_called_once()

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison._compare_pair")
    async def test_two_candidates_b_wins(self, mock_compare):
        mock_compare.return_value = "B"
        result = await vision_pair_select([b"img_a", b"img_b"], "criteria")
        assert result == 1

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison._compare_pair")
    async def test_three_candidates_tournament(self, mock_compare, fake_candidates):
        # First comparison: 0 vs 1 -> B wins (idx 1)
        # Second comparison: 1 vs 2 -> A wins (idx 1 stays)
        mock_compare.side_effect = ["B", "A"]
        result = await vision_pair_select(fake_candidates, "criteria")
        assert result == 1
        assert mock_compare.call_count == 2

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison._compare_pair")
    async def test_three_candidates_last_wins(self, mock_compare, fake_candidates):
        # First comparison: 0 vs 1 -> A wins (idx 0)
        # Second comparison: 0 vs 2 -> B wins (idx 2)
        mock_compare.side_effect = ["A", "B"]
        result = await vision_pair_select(fake_candidates, "criteria")
        assert result == 2

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison._compare_pair")
    async def test_fallback_on_exception(self, mock_compare):
        mock_compare.side_effect = Exception("LLM error")
        result = await vision_pair_select([b"a", b"b"], "criteria")
        assert result == 0


class TestComparePair:
    """Test individual pair comparison."""

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison.invoke")
    async def test_returns_a(self, mock_invoke):
        mock_invoke.return_value = AIMessage(content="A")

        from workflows.shared.llm_utils import ModelTier

        result = await _compare_pair(b"img_a", b"img_b", "criteria", ModelTier.SONNET)
        assert result == "A"

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison.invoke")
    async def test_returns_b(self, mock_invoke):
        mock_invoke.return_value = AIMessage(content="B")

        from workflows.shared.llm_utils import ModelTier

        result = await _compare_pair(b"img_a", b"img_b", "criteria", ModelTier.SONNET)
        assert result == "B"

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison.invoke")
    async def test_ambiguous_response_defaults_to_a(self, mock_invoke):
        mock_invoke.return_value = AIMessage(content="I think image 1 is better")

        from workflows.shared.llm_utils import ModelTier

        result = await _compare_pair(b"img_a", b"img_b", "criteria", ModelTier.SONNET)
        assert result == "A"
