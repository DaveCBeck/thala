"""Module-based logging with run-based rotation.

This package provides per-module log files with automatic rotation at
run boundaries (task queue dispatch or test execution).

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
    - logs/lit-review.log, logs/supervision.log, etc. (per-module)
    - logs/run-3p.log (all third-party libraries)
    - logs/*.previous.log (previous run's logs)
"""

from core.logging.handlers import ModuleDispatchHandler, ThirdPartyHandler
from core.logging.run_manager import (
    MODULE_TO_LOG,
    end_run,
    get_current_run_id,
    module_to_log_name,
    start_run,
)

__all__ = [
    "start_run",
    "end_run",
    "get_current_run_id",
    "module_to_log_name",
    "ModuleDispatchHandler",
    "ThirdPartyHandler",
    "MODULE_TO_LOG",
]
