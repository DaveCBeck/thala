"""Module-based logging with date-stamped files.

This package provides per-module log files using date-stamped filenames
for safe parallel workflow execution.

Usage:
    # At run entry points (task_queue, tests):
    from core.logging import start_run, end_run

    start_run("task-123")
    try:
        # ... do work ...
    finally:
        end_run()

    # In modules (unchanged pattern):
    import logging
    logger = logging.getLogger(__name__)
    logger.info("This goes to the appropriate module log file")

Log files are created in logs/:
    - logs/lit-review.2026-02-10.log, logs/supervision.2026-02-10.log, etc.
    - logs/run-3p.2026-02-10.log (all third-party libraries)
    - Old dated files are cleaned up after THALA_LOG_RETENTION_DAYS (default 7)
"""

from core.logging.handlers import (
    ModuleDispatchHandler,
    RunContextFormatter,
    ThirdPartyHandler,
)
from core.logging.run_manager import (
    MODULE_TO_LOG,
    end_run,
    get_current_run_id,
    module_to_log_name,
    start_run,
)

__all__ = [
    # Run lifecycle
    "start_run",
    "end_run",
    # Handlers (for config.py)
    "ModuleDispatchHandler",
    "ThirdPartyHandler",
    # Formatter
    "RunContextFormatter",
]
