"""Tests for image_utils multi-candidate Imagen generation (A3)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestImagenTimeout:
    """Test that Imagen API calls time out instead of hanging forever."""

    @pytest.mark.asyncio
    @patch("workflows.shared.image_utils._get_genai_client")
    @patch("workflows.shared.imagen_prompts.structure_brief_for_imagen", new_callable=AsyncMock)
    @patch("core.task_queue.rate_limits.get_imagen_semaphore")
    async def test_generate_article_header_timeout_returns_none(
        self, mock_sem, mock_structure, mock_client
    ):
        """A hung generate_images call should timeout and return (None, prompt)."""
        from workflows.shared.image_utils import generate_article_header

        mock_sem.return_value = asyncio.Semaphore(1)
        mock_structure.return_value = "structured prompt"

        # Simulate a call that never returns
        never_resolves: asyncio.Future = asyncio.Future()
        mock_client.return_value.aio.models.generate_images = MagicMock(
            return_value=never_resolves
        )

        with patch("workflows.shared.image_utils.GOOGLE_API_TIMEOUT", 0.05):
            result_bytes, result_prompt = await generate_article_header(
                title="Test", content="", custom_prompt="a brief"
            )

        assert result_bytes is None
        assert result_prompt == "structured prompt"

    @pytest.mark.asyncio
    @patch("workflows.shared.image_utils._get_genai_client")
    @patch("core.task_queue.rate_limits.get_imagen_semaphore")
    async def test_generate_diagram_timeout_returns_none(self, mock_sem, mock_client):
        """A hung generate_content call should timeout and return (None, prompt)."""
        from workflows.shared.image_utils import generate_diagram_image

        mock_sem.return_value = asyncio.Semaphore(1)

        # Simulate a call that never returns
        never_resolves: asyncio.Future = asyncio.Future()
        mock_client.return_value.aio.models.generate_content = MagicMock(
            return_value=never_resolves
        )

        with patch("workflows.shared.image_utils.GOOGLE_API_TIMEOUT", 0.05):
            result_bytes, result_prompt = await generate_diagram_image(brief="test diagram")

        assert result_bytes is None
        assert result_prompt is not None  # prompt is returned even on failure
