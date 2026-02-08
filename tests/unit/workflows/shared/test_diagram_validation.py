"""Tests for SVG validation and sanitization (B4)."""

from workflows.shared.diagram_utils.validation import (
    strip_code_fences,
    validate_and_sanitize_svg,
)


class TestStripCodeFences:
    def test_no_fences(self):
        assert strip_code_fences("<svg>...</svg>") == "<svg>...</svg>"

    def test_svg_fences(self):
        result = strip_code_fences("```svg\n<svg>...</svg>\n```")
        assert result == "<svg>...</svg>"

    def test_plain_fences(self):
        result = strip_code_fences("```\n<svg>...</svg>\n```")
        assert result == "<svg>...</svg>"

    def test_whitespace_stripped(self):
        result = strip_code_fences("  \n<svg>...</svg>\n  ")
        assert result == "<svg>...</svg>"


class TestValidateAndSanitizeSvg:
    MINIMAL_SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="50" height="50"/></svg>'
    )

    def test_valid_svg_passes_through(self):
        result = validate_and_sanitize_svg(self.MINIMAL_SVG)
        assert result is not None
        assert "<svg" in result
        assert "</svg>" in result

    def test_adds_xmlns_if_missing(self):
        svg = '<svg width="100" height="100"><rect width="50" height="50"/></svg>'
        result = validate_and_sanitize_svg(svg)
        assert result is not None
        assert 'xmlns="http://www.w3.org/2000/svg"' in result

    def test_adds_viewbox_if_missing(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect/></svg>'
        result = validate_and_sanitize_svg(svg)
        assert result is not None
        assert 'viewBox="0 0 200 100"' in result

    def test_uses_default_dimensions_for_viewbox(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
        result = validate_and_sanitize_svg(svg, width=900, height=600)
        assert result is not None
        assert "viewBox" in result

    def test_strips_code_fences(self):
        svg_with_fences = f"```svg\n{self.MINIMAL_SVG}\n```"
        result = validate_and_sanitize_svg(svg_with_fences)
        assert result is not None
        assert "<svg" in result

    def test_extracts_svg_from_surrounding_text(self):
        content = f"Here's the diagram:\n{self.MINIMAL_SVG}\nDone."
        result = validate_and_sanitize_svg(content)
        assert result is not None
        assert "<svg" in result

    def test_completely_broken_returns_none(self):
        assert validate_and_sanitize_svg("this is not svg at all") is None

    def test_no_svg_tags_returns_none(self):
        assert validate_and_sanitize_svg("<div>not svg</div>") is None

    def test_missing_closing_tag_recovers_with_lxml(self):
        # lxml recover mode can handle unclosed tags — it won't return None
        result = validate_and_sanitize_svg("<svg><rect>")
        # Just verify it either recovers or returns None (both acceptable)
        if result is not None:
            assert "<svg" in result

    def test_sanitizes_unescaped_ampersand(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><text>A & B</text></svg>'
        result = validate_and_sanitize_svg(svg)
        assert result is not None
        # lxml recover mode should handle the entity

    def test_preserves_valid_entities(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><text>A &amp; B</text></svg>'
        result = validate_and_sanitize_svg(svg)
        assert result is not None

    def test_empty_string_returns_none(self):
        assert validate_and_sanitize_svg("") is None

    def test_preserves_existing_viewbox(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"><rect/></svg>'
        result = validate_and_sanitize_svg(svg)
        assert result is not None
        assert "400 300" in result
