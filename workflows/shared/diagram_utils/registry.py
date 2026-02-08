"""Diagram engine availability registry.

Checks engine availability once at first access, not per-call.
Used by the diagram routing logic to skip unavailable engines
gracefully rather than failing at runtime.
"""

import logging
import shutil

logger = logging.getLogger(__name__)

_available_engines: set[str] = set()
_checked = False


def get_available_engines() -> set[str]:
    """Return set of available diagram engines. Checked once."""
    global _checked
    if not _checked:
        _available_engines.add("svg")  # always available (cairosvg)
        try:
            import mmdc  # noqa: F401

            _available_engines.add("mermaid")
            logger.debug("Mermaid engine available (mmdc)")
        except ImportError:
            logger.info("Mermaid engine unavailable (mmdc not installed)")

        if shutil.which("dot"):
            _available_engines.add("graphviz")
            logger.debug("Graphviz engine available (dot binary found)")
        else:
            logger.info("Graphviz engine unavailable (dot binary not found)")

        _checked = True
        logger.info(f"Available diagram engines: {_available_engines}")
    return _available_engines


def is_engine_available(engine: str) -> bool:
    """Check if a specific engine is available."""
    return engine in get_available_engines()


def reset_registry() -> None:
    """Reset the registry for testing."""
    global _checked
    _available_engines.clear()
    _checked = False


__all__ = [
    "get_available_engines",
    "is_engine_available",
    "reset_registry",
]
