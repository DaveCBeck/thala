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


def _rotate_log(log_dir: Path, base_name: str, keep: int = 4) -> None:
    """Rotate log file: current -> previous/name.1.log, shift .1->.2, etc.

    Args:
        log_dir: Directory containing the log file
        base_name: Name of the log file (e.g., "thala.log")
        keep: Number of previous versions to keep (default: 4)
    """
    previous_dir = log_dir / "previous"
    current = log_dir / base_name

    if not current.exists():
        return

    previous_dir.mkdir(exist_ok=True)

    # Parse stem and suffix
    if "." in base_name:
        stem, suffix = base_name.rsplit(".", 1)
    else:
        stem, suffix = base_name, "log"

    # Delete oldest, shift others (.4 deleted, .3->.4, .2->.3, .1->.2)
    for i in range(keep, 0, -1):
        src = previous_dir / f"{stem}.{i}.{suffix}"
        dst = previous_dir / f"{stem}.{i + 1}.{suffix}"
        if i == keep and src.exists():
            src.unlink()
        elif src.exists():
            src.rename(dst)

    # Move current to .1
    current.rename(previous_dir / f"{stem}.1.{suffix}")


def configure_logging(name: str = "thala") -> Path:
    """Configure application-wide logging with console and file handlers.

    Sets up dual logging:
    - Console: Compact format, defaults to WARNING level
    - File: Detailed format with timestamps, defaults to INFO level

    Third-party library logs are segregated to a separate file to reduce noise.

    Args:
        name: Base name for log files (default: "thala").
              Creates {name}.log and {name}-3p.log with stable names.

    Environment variables:
        THALA_LOG_LEVEL_CONSOLE: Console log level (default: WARNING)
        THALA_LOG_LEVEL_FILE: File log level (default: INFO)
        THALA_LOG_DIR: Directory for log files (default: ./logs/)

    Log files use stable names (e.g., thala.log). Previous versions are
    rotated to logs/previous/ (keeps 4 previous versions).

    Returns:
        Path to the main log file

    Note:
        Safe to call multiple times (idempotent). Returns log path on
        subsequent calls without reconfiguring.
    """
    global _logging_configured

    # Determine log directory
    log_dir = Path(os.getenv("THALA_LOG_DIR", _get_project_root() / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    if _logging_configured:
        # Return existing log file path
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                return Path(handler.baseFilename)
        return log_dir / f"{name}.log"

    # Get configuration from environment
    console_level = os.getenv("THALA_LOG_LEVEL_CONSOLE", DEFAULT_CONSOLE_LEVEL).upper()
    file_level = os.getenv("THALA_LOG_LEVEL_FILE", DEFAULT_FILE_LEVEL).upper()

    # Rotate existing log files to previous/ subdirectory (keep 4 previous)
    _rotate_log(log_dir, f"{name}.log", keep=4)
    _rotate_log(log_dir, f"{name}-3p.log", keep=4)

    # Use stable log file names
    main_log_file = log_dir / f"{name}.log"
    third_party_log_file = log_dir / f"{name}-3p.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything, handlers filter
    root_logger.handlers.clear()

    # Formatters
    console_formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler (for thala code only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level, logging.WARNING))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Main file handler (for thala code)
    main_file_handler = logging.FileHandler(main_log_file, mode="w", encoding="utf-8")
    main_file_handler.setLevel(getattr(logging, file_level, logging.INFO))
    main_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(main_file_handler)

    # Third-party file handler
    third_party_handler = logging.FileHandler(
        third_party_log_file, mode="w", encoding="utf-8"
    )
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
    logger.debug(f"Main log: {main_log_file}")
    logger.debug(f"Third-party log: {third_party_log_file}")

    return main_log_file


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
