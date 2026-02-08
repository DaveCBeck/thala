"""Tests for Graphviz diagram generation (B2)."""

from unittest.mock import MagicMock, patch

import pytest

from workflows.shared.diagram_utils.graphviz_engine import (
    generate_graphviz_diagram,
    generate_graphviz_with_selection,
)
from workflows.shared.diagram_utils.schemas import DiagramConfig, DiagramResult


@pytest.fixture
def config():
    return DiagramConfig(width=800, height=600)


class TestGenerateGraphvizDiagram:
    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.graphviz_engine.invoke")
    @patch("workflows.shared.diagram_utils.graphviz_engine._render_dot_to_png")
    async def test_success_flow(self, mock_render, mock_invoke, config):
        mock_response = MagicMock()
        mock_response.content = "digraph { A -> B }"
        mock_invoke.return_value = mock_response
        mock_render.return_value = (b"fake_png", None)

        result = await generate_graphviz_diagram("test analysis", config)
        assert result.success is True
        assert result.png_bytes == b"fake_png"

    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.graphviz_engine.invoke")
    async def test_llm_failure(self, mock_invoke, config):
        mock_invoke.side_effect = Exception("LLM error")
        result = await generate_graphviz_diagram("test", config)
        assert result.success is False

    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.graphviz_engine.invoke")
    @patch("workflows.shared.diagram_utils.graphviz_engine._render_dot_to_png")
    async def test_repair_on_render_failure(self, mock_render, mock_invoke, config):
        gen_response = MagicMock()
        gen_response.content = "bad dot code"
        repair_response = MagicMock()
        repair_response.content = "digraph { A -> B }"
        mock_invoke.side_effect = [gen_response, repair_response]

        # First render fails, second succeeds
        mock_render.side_effect = [
            (None, "syntax error"),
            (b"png_bytes", None),
        ]

        result = await generate_graphviz_diagram("test", config)
        assert result.success is True
        assert mock_invoke.call_count == 2

    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.graphviz_engine.invoke")
    @patch("workflows.shared.diagram_utils.graphviz_engine._render_dot_to_png")
    async def test_all_repairs_fail(self, mock_render, mock_invoke, config):
        mock_response = MagicMock()
        mock_response.content = "broken"
        mock_invoke.return_value = mock_response
        mock_render.return_value = (None, "still broken")

        result = await generate_graphviz_diagram("test", config)
        assert result.success is False


class TestGenerateGraphvizWithSelection:
    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.graphviz_engine.generate_graphviz_diagram")
    async def test_all_fail(self, mock_gen, config):
        mock_gen.return_value = DiagramResult(
            svg_bytes=None,
            png_bytes=None,
            analysis=None,
            overlap_check=None,
            generation_attempts=1,
            success=False,
            error="fail",
        )
        result = await generate_graphviz_with_selection("test", config, num_candidates=3)
        assert result.success is False

    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.graphviz_engine.generate_graphviz_diagram")
    async def test_single_success_skips_vision(self, mock_gen, config):
        success = DiagramResult(
            svg_bytes=None,
            png_bytes=b"png",
            analysis=None,
            overlap_check=None,
            generation_attempts=1,
            success=True,
            source_code="dot",
        )
        failure = DiagramResult(
            svg_bytes=None,
            png_bytes=None,
            analysis=None,
            overlap_check=None,
            generation_attempts=1,
            success=False,
            error="fail",
        )
        mock_gen.side_effect = [success, failure, failure]

        result = await generate_graphviz_with_selection("test", config, num_candidates=3)
        assert result.success is True
        assert result.png_bytes == b"png"

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison.vision_pair_select")
    @patch("workflows.shared.diagram_utils.graphviz_engine.generate_graphviz_diagram")
    async def test_vision_selects_best(self, mock_gen, mock_vision, config):
        results = [
            DiagramResult(
                svg_bytes=None,
                png_bytes=b"p1",
                analysis=None,
                overlap_check=None,
                generation_attempts=1,
                success=True,
                source_code="d1",
            ),
            DiagramResult(
                svg_bytes=None,
                png_bytes=b"p2",
                analysis=None,
                overlap_check=None,
                generation_attempts=1,
                success=True,
                source_code="d2",
            ),
            DiagramResult(
                svg_bytes=None,
                png_bytes=b"p3",
                analysis=None,
                overlap_check=None,
                generation_attempts=1,
                success=True,
                source_code="d3",
            ),
        ]
        mock_gen.side_effect = results
        mock_vision.return_value = 2

        result = await generate_graphviz_with_selection("test", config, num_candidates=3)
        assert result.success is True
        assert result.png_bytes == b"p3"
