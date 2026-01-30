"""Run-based log rotation manager.

Provides run lifecycle management for module-based logging. A "run" is a
logical unit of work (task queue dispatch or test execution) that triggers
log rotation on first write to each module's log file.

Usage:
    from core.logging import start_run, end_run

    start_run("task-abc123")  # Triggers rotation on first log to each module
    try:
        # ... do work ...
    finally:
        end_run()
"""

from contextvars import ContextVar

# Both must be ContextVars for async safety - prevents state leakage between
# concurrent async runs
_current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)
_rotated_this_run: ContextVar[set[str] | None] = ContextVar("rotated_this_run", default=None)

# Cache module-to-log resolution (populated on first access per module name)
# This is a global cache shared across runs - module names don't change
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

    Triggers log rotation on first log message to each module within this run.
    Safe to call multiple times - subsequent calls reset the rotation tracking.

    Args:
        run_id: Unique identifier for this run (e.g., task_id, test name)
    """
    _current_run_id.set(run_id)
    _rotated_this_run.set(set())  # Fresh set per async context


def end_run() -> None:
    """Signal end of run.

    Best-effort cleanup - may not be called on crash. Rotation is triggered
    by start_run(), so missing end_run() calls don't affect correctness.
    """
    _current_run_id.set(None)
    _rotated_this_run.set(None)


def get_current_run_id() -> str | None:
    """Get the current run ID, if any."""
    return _current_run_id.get()


def should_rotate(log_name: str) -> bool:
    """Check if rotation is needed for this log file.

    Returns True if:
    1. We're in a run (start_run() was called)
    2. This log file hasn't been rotated yet in this run

    Also marks the log as rotated to prevent duplicate rotations.

    Args:
        log_name: The log file name (without .log extension)

    Returns:
        True if rotation should happen, False otherwise
    """
    run_id = _current_run_id.get()
    rotated = _rotated_this_run.get()

    if run_id is None or rotated is None:
        return False

    if log_name in rotated:
        return False

    # Mark as rotated before returning
    rotated.add(log_name)
    return True


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

    Args:
        module_name: The __name__ of the module

    Returns:
        Log file name, or "misc" if no prefix matches
    """
    for prefix in _SORTED_PREFIXES:
        if module_name.startswith(prefix):
            return MODULE_TO_LOG[prefix]
    return "misc"
