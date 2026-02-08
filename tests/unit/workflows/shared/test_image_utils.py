"""Tests for image_utils multi-candidate Imagen generation (A3)."""

from unittest.mock import AsyncMock, patch

import pytest


class TestSelectBestImagenCandidate:
    """Test _select_best_imagen_candidate vision selection."""

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison.vision_pair_select", new_callable=AsyncMock)
    async def test_selects_best_via_vision(self, mock_select):
        from workflows.shared.image_utils import _select_best_imagen_candidate

        mock_select.return_value = 2
        candidates = [b"img0", b"img1", b"img2"]

        result = await _select_best_imagen_candidate(candidates, "prompt", "brief")
        assert result == b"img2"
        mock_select.assert_called_once_with(candidates, "brief")

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison.vision_pair_select", new_callable=AsyncMock)
    async def test_uses_prompt_when_no_brief(self, mock_select):
        from workflows.shared.image_utils import _select_best_imagen_candidate

        mock_select.return_value = 0
        candidates = [b"img0", b"img1"]

        result = await _select_best_imagen_candidate(candidates, "the prompt", "")
        assert result == b"img0"
        mock_select.assert_called_once_with(candidates, "the prompt")

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison.vision_pair_select", new_callable=AsyncMock)
    async def test_fallback_on_error(self, mock_select):
        from workflows.shared.image_utils import _select_best_imagen_candidate

        mock_select.side_effect = Exception("Vision model unavailable")
        candidates = [b"img0", b"img1", b"img2"]

        result = await _select_best_imagen_candidate(candidates, "prompt", "brief")
        assert result == b"img0"  # Falls back to first
