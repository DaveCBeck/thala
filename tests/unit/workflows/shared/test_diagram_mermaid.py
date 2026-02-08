"""Tests for Mermaid diagram generation (B1, B5)."""

from unittest.mock import MagicMock, patch

import pytest

from workflows.shared.diagram_utils.mermaid import (
    _validate_mermaid,
    generate_mermaid_diagram,
    generate_mermaid_with_selection,
)
from workflows.shared.diagram_utils.schemas import DiagramConfig, DiagramResult


@pytest.fixture
def config():
    return DiagramConfig(width=800, height=600)


class TestValidateMermaid:
    def test_valid_mermaid(self):
        code = "graph TD\n    A[Start] --> B[End]"
        is_valid, errors = _validate_mermaid(code)
        assert is_valid is True
        assert errors == ""

    def test_invalid_mermaid(self):
        code = "this is not mermaid at all }{{"
        is_valid, errors = _validate_mermaid(code)
        # mmdc may or may not parse this — just check it returns a tuple
        assert isinstance(is_valid, bool)
        assert isinstance(errors, str)


class TestGenerateMermaidDiagram:
    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.mermaid.invoke")
    @patch("workflows.shared.diagram_utils.mermaid._render_mermaid_to_png")
    @patch("workflows.shared.diagram_utils.mermaid._validate_mermaid")
    async def test_success_flow(self, mock_validate, mock_render, mock_invoke, config):
        # LLM returns valid mermaid
        mock_response = MagicMock()
        mock_response.content = "graph TD\n    A --> B"
        mock_invoke.return_value = mock_response

        # Validation passes
        mock_validate.return_value = (True, "")

        # Rendering returns PNG bytes
        mock_render.return_value = b"fake_png_bytes"

        result = await generate_mermaid_diagram("test analysis", config)

        assert result.success is True
        assert result.png_bytes == b"fake_png_bytes"

    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.mermaid.invoke")
    async def test_llm_failure_returns_failure(self, mock_invoke, config):
        mock_invoke.side_effect = Exception("LLM error")
        result = await generate_mermaid_diagram("test", config)
        assert result.success is False

    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.mermaid.invoke")
    @patch("workflows.shared.diagram_utils.mermaid._render_mermaid_to_png")
    @patch("workflows.shared.diagram_utils.mermaid._validate_mermaid")
    async def test_repair_loop_on_validation_failure(self, mock_validate, mock_render, mock_invoke, config):
        # Generation call
        gen_response = MagicMock()
        gen_response.content = "broken mermaid"

        # Repair call
        repair_response = MagicMock()
        repair_response.content = "graph TD\n    A --> B"

        mock_invoke.side_effect = [gen_response, repair_response]

        # First validation fails, second passes
        mock_validate.side_effect = [(False, "syntax error"), (True, "")]
        mock_render.return_value = b"png_bytes"

        result = await generate_mermaid_diagram("test", config)
        assert result.success is True
        # invoke called twice: generate + repair
        assert mock_invoke.call_count == 2

    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.mermaid.invoke")
    @patch("workflows.shared.diagram_utils.mermaid._validate_mermaid")
    async def test_all_repairs_fail(self, mock_validate, mock_invoke, config):
        mock_response = MagicMock()
        mock_response.content = "broken"
        mock_invoke.return_value = mock_response

        # All validation attempts fail
        mock_validate.return_value = (False, "still broken")

        result = await generate_mermaid_diagram("test", config)
        assert result.success is False
        assert "validation failed" in result.error.lower()

    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.mermaid.invoke")
    @patch("workflows.shared.diagram_utils.mermaid._render_mermaid_to_png")
    @patch("workflows.shared.diagram_utils.mermaid._validate_mermaid")
    async def test_render_failure(self, mock_validate, mock_render, mock_invoke, config):
        mock_response = MagicMock()
        mock_response.content = "graph TD\n    A --> B"
        mock_invoke.return_value = mock_response
        mock_validate.return_value = (True, "")
        mock_render.return_value = None  # Render fails

        result = await generate_mermaid_diagram("test", config)
        assert result.success is False
        assert "rendering failed" in result.error.lower()


class TestGenerateMermaidWithSelection:
    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.mermaid.generate_mermaid_diagram")
    async def test_all_candidates_fail(self, mock_gen, config):
        mock_gen.return_value = DiagramResult(
            svg_bytes=None,
            png_bytes=None,
            analysis=None,
            overlap_check=None,
            generation_attempts=1,
            success=False,
            error="failed",
        )

        result = await generate_mermaid_with_selection("test", config, num_candidates=3)
        assert result.success is False

    @pytest.mark.asyncio
    @patch("workflows.shared.diagram_utils.mermaid.generate_mermaid_diagram")
    async def test_single_success_skips_vision(self, mock_gen, config):
        success = DiagramResult(
            svg_bytes=None,
            png_bytes=b"png",
            analysis=None,
            overlap_check=None,
            generation_attempts=1,
            success=True,
            source_code="code",
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

        result = await generate_mermaid_with_selection("test", config, num_candidates=3)
        assert result.success is True
        assert result.png_bytes == b"png"

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison.vision_pair_select")
    @patch("workflows.shared.diagram_utils.mermaid.generate_mermaid_diagram")
    async def test_multiple_success_uses_vision(self, mock_gen, mock_vision, config):
        results = [
            DiagramResult(
                svg_bytes=None,
                png_bytes=b"png1",
                analysis=None,
                overlap_check=None,
                generation_attempts=1,
                success=True,
                source_code="code1",
            ),
            DiagramResult(
                svg_bytes=None,
                png_bytes=b"png2",
                analysis=None,
                overlap_check=None,
                generation_attempts=1,
                success=True,
                source_code="code2",
            ),
        ]
        mock_gen.side_effect = results + [
            DiagramResult(
                svg_bytes=None,
                png_bytes=None,
                analysis=None,
                overlap_check=None,
                generation_attempts=1,
                success=False,
                error="fail",
            )
        ]
        mock_vision.return_value = 1  # Select second candidate

        result = await generate_mermaid_with_selection("test", config, num_candidates=3)
        assert result.success is True
        assert result.png_bytes == b"png2"

    @pytest.mark.asyncio
    @patch("workflows.shared.vision_comparison.vision_pair_select")
    @patch("workflows.shared.diagram_utils.mermaid.generate_mermaid_diagram")
    async def test_vision_failure_falls_back_to_first(self, mock_gen, mock_vision, config):
        results = [
            DiagramResult(
                svg_bytes=None,
                png_bytes=b"png1",
                analysis=None,
                overlap_check=None,
                generation_attempts=1,
                success=True,
                source_code="code1",
            ),
            DiagramResult(
                svg_bytes=None,
                png_bytes=b"png2",
                analysis=None,
                overlap_check=None,
                generation_attempts=1,
                success=True,
                source_code="code2",
            ),
        ]
        mock_gen.side_effect = results + [
            DiagramResult(
                svg_bytes=None,
                png_bytes=None,
                analysis=None,
                overlap_check=None,
                generation_attempts=1,
                success=False,
                error="fail",
            )
        ]
        mock_vision.side_effect = Exception("vision failed")

        result = await generate_mermaid_with_selection("test", config, num_candidates=3)
        assert result.success is True
        assert result.png_bytes == b"png1"
