"""Run context manager for date-based logging.

Provides run lifecycle management for module-based logging. A "run" is a
logical unit of work (task queue dispatch or test execution). The run ID
is embedded in log lines via RunContextFormatter so parallel tasks are
distinguishable.

Usage:
    from core.logging import start_run, end_run

    start_run("task-abc123")
    try:
        # ... do work ...
    finally:
        end_run()
"""

from contextvars import ContextVar

_current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)

# Cache module-to-log resolution (populated on first access per module name)
# This is a global cache shared across runs - module names don't change
# IMPORTANT: This cache assumes logger names follow the standard __name__ pattern.
# Do NOT create loggers with dynamic names (e.g., f"module.{task_id}") as the cache
# is unbounded. Module names are finite and determined by the codebase structure.
_module_log_cache: dict[str, str] = {}

# Mapping from module path prefixes to log file names
# Uses longest-prefix-match to resolve module paths to log names
# Unmapped modules go to "misc.log"
MODULE_TO_LOG = {
    # Workflows - research
    "workflows.research.academic_lit_review": "lit-review",
    "workflows.research.book_finding": "book-finding",
    "workflows.research.web_research": "web-research",
    # Workflows - enhance
    "workflows.enhance.supervision": "supervision",
    "workflows.enhance.editing": "editing",
    "workflows.enhance.fact_check": "fact-check",
    # Workflows - output
    "workflows.output.evening_reads": "evening-reads",
    "workflows.output.illustrate": "illustrate",
    # Workflows - other
    "workflows.document_processing": "doc-processing",
    "workflows.shared": "workflows-shared",
    "workflows.wrappers": "workflows-wrappers",
    # Core modules (intentionally grouped)
    "core.stores": "stores",
    "core.task_queue": "task-queue",
    "core.scraping": "scraping",
    "core.images": "images",
    "core.utils": "utils",
    "core.config": "config",
    "core.logging": "logging-internal",
    # Other top-level modules
    "langchain_tools": "langchain-tools",
    "testing": "testing",
}

# Pre-sorted prefixes by length (longest first) for efficient matching
_SORTED_PREFIXES = sorted(MODULE_TO_LOG.keys(), key=len, reverse=True)


def start_run(run_id: str) -> None:
    """Signal start of new run.

    Sets the run ID in the current async context. The run ID is embedded in
    log lines by RunContextFormatter.

    Args:
        run_id: Unique identifier for this run (e.g., task_id, test name)
    """
    _current_run_id.set(run_id)


def end_run() -> None:
    """Signal end of run.

    Clears the run ID from the current async context.
    """
    _current_run_id.set(None)


def get_current_run_id() -> str | None:
    """Get the current run ID, if any."""
    return _current_run_id.get()


def module_to_log_name(module_name: str) -> str:
    """Resolve module path to log filename.

    Uses longest-prefix-match against MODULE_TO_LOG mapping.
    Results are cached for performance.

    Args:
        module_name: The __name__ of the module (e.g., "core.stores.elasticsearch.client")

    Returns:
        Log file name without extension (e.g., "stores")
    """
    if module_name not in _module_log_cache:
        _module_log_cache[module_name] = _compute_log_name(module_name)
    return _module_log_cache[module_name]


def _compute_log_name(module_name: str) -> str:
    """Find longest matching prefix in MODULE_TO_LOG.

    SECURITY: Unmapped modules fall back to "misc" which prevents
    arbitrary file creation via malicious logger names. Never use
    module_name directly in file paths without sanitization.

    Args:
        module_name: The __name__ of the module

    Returns:
        Log file name, or "misc" if no prefix matches
    """
    for prefix in _SORTED_PREFIXES:
        if module_name.startswith(prefix):
            return MODULE_TO_LOG[prefix]
    return "misc"
