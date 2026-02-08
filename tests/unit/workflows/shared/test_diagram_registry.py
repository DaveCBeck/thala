"""Tests for diagram engine availability registry (R5)."""

from unittest.mock import patch

from workflows.shared.diagram_utils.registry import (
    get_available_engines,
    is_engine_available,
    reset_registry,
)


class TestEngineRegistry:
    def setup_method(self):
        reset_registry()

    def test_svg_always_available(self):
        engines = get_available_engines()
        assert "svg" in engines

    def test_caches_result(self):
        engines1 = get_available_engines()
        engines2 = get_available_engines()
        assert engines1 is engines2

    @patch("workflows.shared.diagram_utils.registry.shutil.which", return_value=None)
    def test_graphviz_unavailable_when_no_dot(self, mock_which):
        reset_registry()
        # Block mmdc import too
        with patch.dict("sys.modules", {"mmdc": None}):
            import importlib

            import workflows.shared.diagram_utils.registry as reg

            importlib.reload(reg)
            reg.reset_registry()

            # Mock import failure for mmdc
            original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

            def mock_import(name, *args, **kwargs):
                if name == "mmdc":
                    raise ImportError("mocked")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                engines = reg.get_available_engines()
                assert "graphviz" not in engines

    def test_reset_clears_cache(self):
        get_available_engines()
        reset_registry()
        # After reset, should re-check on next call
        engines = get_available_engines()
        assert "svg" in engines

    def test_is_engine_available_delegates(self):
        assert is_engine_available("svg") is True
        # The real environment has mmdc installed
        assert is_engine_available("mermaid") is True

    def test_mermaid_available_when_mmdc_installed(self):
        # mmdc is installed in our test env
        reset_registry()
        assert is_engine_available("mermaid") is True

    def test_nonexistent_engine(self):
        assert is_engine_available("fictional_engine") is False
