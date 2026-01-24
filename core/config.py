"""Thala configuration and environment setup.

This module provides centralized configuration for the Thala system,
including development mode detection, LangSmith tracing setup, and logging.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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


def _cleanup_old_logs(log_dir: Path, pattern: str, keep: int = 5) -> None:
    """Remove old log files, keeping the most recent ones.

    Args:
        log_dir: Directory containing log files
        pattern: Glob pattern for log files (e.g., "thala_*.log")
        keep: Number of recent files to keep
    """
    log_files = sorted(
        log_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_file in log_files[keep:]:
        old_file.unlink(missing_ok=True)


def configure_logging(name: str = "thala") -> Path:
    """Configure application-wide logging with console and file handlers.

    Sets up dual logging:
    - Console: Compact format, defaults to WARNING level
    - File: Detailed format with timestamps, defaults to INFO level

    Third-party library logs are segregated to a separate file to reduce noise.

    Args:
        name: Base name for log files (default: "thala").
              Creates {name}_{datetime}.log and {name}-3p_{datetime}.log

    Environment variables:
        THALA_LOG_LEVEL_CONSOLE: Console log level (default: WARNING)
        THALA_LOG_LEVEL_FILE: File log level (default: INFO)
        THALA_LOG_DIR: Directory for log files (default: ./logs/)

    Log files use datetime format: {name}_YYYYMMDD_HHMMSS.log
    Keeps 5 most recent log files per name.

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

    # Clean up old log files (keep 5 most recent of each type)
    _cleanup_old_logs(log_dir, f"{name}_*.log", keep=5)
    _cleanup_old_logs(log_dir, f"{name}-3p_*.log", keep=5)

    # Create log files with datetime stamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_file = log_dir / f"{name}_{timestamp}.log"
    third_party_log_file = log_dir / f"{name}-3p_{timestamp}.log"

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
