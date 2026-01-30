"""Thala configuration and environment setup.

This module provides centralized configuration for the Thala system,
including development mode detection, LangSmith tracing setup, and logging.
"""

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# LangSmith trace size limit (bytes) - API limit is 20MB, use 15MB for safety margin
LANGSMITH_MAX_TRACE_SIZE = int(os.getenv("LANGSMITH_MAX_TRACE_SIZE", str(15 * 1024 * 1024)))

# Logging defaults
DEFAULT_CONSOLE_LEVEL = "WARNING"
DEFAULT_FILE_LEVEL = "INFO"

# Third-party loggers to segregate into separate log file
# Note: langchain/langgraph are kept in main logs for workflow visibility
THIRD_PARTY_LOGGERS = [
    "httpx",
    "httpcore",
    "urllib3",
    "asyncio",
    "elasticsearch",
    "chromadb",
    "openai",
    "anthropic",
    "httpx._client",
    "hpack",
    "charset_normalizer",
    "filelock",
    "fsspec",
    "huggingface_hub",
    "numba",
    "voyage",
]

# Loggers that should be set to WARNING to avoid race conditions during cleanup
# httpcore logs DEBUG messages during connection close which can cause reentrant
# call errors when multiple async operations try to log simultaneously
NOISY_LOGGERS = [
    "httpcore",
    "httpcore._trace",
    "httpcore.http11",
    "httpcore.connection",
    "httpcore._async.http11",
    "httpcore._async.connection",
    "httpcore._async.connection_pool",
    "httpcore._sync.http11",
    "httpcore._sync.connection",
    "httpcore._sync.connection_pool",
]

_logging_configured = False


def _get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root (directory containing core/)
    """
    return Path(__file__).parent.parent


def configure_logging(name: str = "thala") -> Path:
    """Configure application-wide logging with console and module-based file handlers.

    Sets up logging with:
    - Console: Compact format, defaults to WARNING level
    - Per-module files: Detailed format, defaults to INFO level
      Each module writes to logs/<module>.log based on MODULE_TO_LOG mapping
    - Third-party: All third-party logs go to logs/run-3p.log

    Log rotation happens automatically on each "run" (task queue dispatch or
    test execution) via start_run(). Each module keeps current + previous log.

    Args:
        name: Unused, kept for backwards compatibility.

    Environment variables:
        THALA_LOG_LEVEL_CONSOLE: Console log level (default: WARNING)
        THALA_LOG_LEVEL_FILE: File log level (default: INFO)
        THALA_LOG_DIR: Directory for log files (default: ./logs/)

    Returns:
        Path to the log directory

    Note:
        Safe to call multiple times (idempotent). Returns log dir on
        subsequent calls without reconfiguring.
    """
    global _logging_configured

    from core.logging import ModuleDispatchHandler, ThirdPartyHandler

    # Determine log directory
    log_dir = Path(os.getenv("THALA_LOG_DIR", _get_project_root() / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    if _logging_configured:
        return log_dir

    # Get configuration from environment
    console_level = os.getenv("THALA_LOG_LEVEL_CONSOLE", DEFAULT_CONSOLE_LEVEL).upper()
    file_level = os.getenv("THALA_LOG_LEVEL_FILE", DEFAULT_FILE_LEVEL).upper()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything, handlers filter
    root_logger.handlers.clear()

    # Formatters
    console_formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler (for thala code only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level, logging.WARNING))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Module-based file handler (routes to per-module log files)
    module_handler = ModuleDispatchHandler(log_dir)
    module_handler.setLevel(getattr(logging, file_level, logging.INFO))
    module_handler.setFormatter(file_formatter)
    root_logger.addHandler(module_handler)

    # Third-party file handler (single run-3p.log for all third-party libs)
    third_party_handler = ThirdPartyHandler(log_dir)
    third_party_handler.setLevel(logging.DEBUG)  # Capture all third-party logs
    third_party_handler.setFormatter(file_formatter)

    # Configure third-party loggers to use separate file only
    for logger_name in THIRD_PARTY_LOGGERS:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.handlers.clear()
        third_party_logger.addHandler(third_party_handler)
        third_party_logger.propagate = False  # Don't send to root logger

    # Suppress noisy loggers that cause race conditions during async cleanup
    # These log DEBUG messages during connection close which can trigger
    # reentrant call errors in FileHandler when multiple operations flush
    for logger_name in NOISY_LOGGERS:
        noisy_logger = logging.getLogger(logger_name)
        noisy_logger.setLevel(logging.WARNING)

    _logging_configured = True

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured: console={console_level}, file={file_level}")
    logger.debug(f"Log directory: {log_dir}")

    return log_dir


def is_dev_mode() -> bool:
    """Check if running in development mode.

    Returns:
        True if THALA_MODE is set to 'dev', False otherwise.
    """
    return os.getenv("THALA_MODE", "prod").lower() == "dev"


def configure_langsmith() -> None:
    """Configure LangSmith tracing based on THALA_MODE.

    When THALA_MODE=dev:
        - Enables LangSmith tracing
        - Sets project to 'thala-dev'

    When THALA_MODE=prod (or unset):
        - Disables LangSmith tracing

    This function is idempotent and safe to call multiple times.
    """
    if is_dev_mode():
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "thala-dev")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"


def truncate_for_trace(data: Any, max_str_len: int = 50000) -> Any:
    """Truncate large strings in data structures for LangSmith tracing.

    Use with @traceable(process_inputs=truncate_for_trace, process_outputs=truncate_for_trace)
    to prevent oversized trace payloads that exceed LangSmith's 20MB limit.

    Args:
        data: Input/output data from a traced function
        max_str_len: Maximum length for string fields (default 50KB)

    Returns:
        Data with large strings truncated
    """
    if isinstance(data, str):
        if len(data) > max_str_len:
            return data[:max_str_len] + f"\n\n[TRUNCATED - {len(data):,} chars total]"
        return data
    elif isinstance(data, dict):
        return {k: truncate_for_trace(v, max_str_len) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_for_trace(item, max_str_len) for item in data]
    elif hasattr(data, "__dict__"):
        # Handle dataclasses/objects - return dict representation
        try:
            return {k: truncate_for_trace(v, max_str_len) for k, v in data.__dict__.items()}
        except Exception:
            return str(data)[:max_str_len] if len(str(data)) > max_str_len else data
    return data
