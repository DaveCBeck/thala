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
    "pypdf",
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
    - Per-module files: Date-stamped files ({module}.{YYYY-MM-DD}.log),
      defaults to INFO level. Each module's log is based on MODULE_TO_LOG mapping.
    - Third-party: All third-party logs go to run-3p.{YYYY-MM-DD}.log

    File handlers use RunContextFormatter to embed [run_id] in each line,
    making parallel tasks distinguishable via grep. Old dated files are
    cleaned up after THALA_LOG_RETENTION_DAYS (default 7).

    Args:
        name: Unused, kept for backwards compatibility.

    Environment variables:
        THALA_LOG_LEVEL_CONSOLE: Console log level (default: WARNING)
        THALA_LOG_LEVEL_FILE: File log level (default: INFO)
        THALA_LOG_DIR: Directory for log files (default: ./logs/)
        THALA_LOG_RETENTION_DAYS: Days to keep dated log files (default: 7)

    Returns:
        Path to the log directory

    Note:
        Safe to call multiple times (idempotent). Returns log dir on
        subsequent calls without reconfiguring.
    """
    global _logging_configured

    from core.logging import ModuleDispatchHandler, RunContextFormatter, ThirdPartyHandler

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
    file_formatter = RunContextFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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


def _strip_bytes_from_trace(data: dict) -> dict:
    """Strip binary data and large base64 strings from LangSmith trace payloads.

    The illustrate workflow stores raw image_bytes in LangGraph state,
    and vision comparison sends base64-encoded images as LLM inputs.
    Both can exceed LangSmith's 20MB payload limit. This replaces bytes
    with a size placeholder and truncates large strings (e.g. base64 data).
    """
    _MAX_STR = 50_000  # 50KB — keeps traces useful without the bulk

    def _walk(obj: Any) -> Any:
        if isinstance(obj, bytes):
            return f"<binary: {len(obj):,} bytes>"
        if isinstance(obj, str):
            if len(obj) > _MAX_STR:
                return obj[:200] + f"... [TRUNCATED - {len(obj):,} chars]"
            return obj
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(v) for v in obj)
        return obj

    return _walk(data)


def configure_langsmith() -> None:
    """Configure LangSmith tracing based on THALA_MODE.

    When THALA_MODE=dev:
        - Enables LangSmith tracing
        - Sets project to 'thala-dev'
        - Pre-initializes the global client with byte-stripping filters
          to prevent oversized trace payloads from image data

    When THALA_MODE=prod (or unset):
        - Disables LangSmith tracing

    This function is idempotent and safe to call multiple times.
    """
    if is_dev_mode():
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "thala-dev")
        # Pre-initialize the global langsmith client with byte-stripping filters.
        # LangGraph serializes full graph state (including image_bytes) into traces;
        # without filtering, the illustrate workflow's ~45MB of image data exceeds
        # LangSmith's 20MB payload limit.
        try:
            from langsmith.run_trees import get_cached_client

            get_cached_client(
                hide_inputs=_strip_bytes_from_trace,
                hide_outputs=_strip_bytes_from_trace,
            )
        except Exception:
            pass  # Non-critical — tracing still works, just may warn on large payloads
    else:
        os.environ["LANGSMITH_TRACING"] = "false"


def truncate_for_trace(data: Any, max_str_len: int = 50000, _depth: int = 0) -> Any:
    """Truncate large strings/bytes in data structures for LangSmith tracing.

    Uses copy-on-write: containers are only copied when a descendant value
    actually needs truncation, avoiding allocation of hundreds of thousands
    of identical dict/list copies on every @traceable call.

    Use with @traceable(process_inputs=truncate_for_trace, process_outputs=truncate_for_trace)
    to prevent oversized trace payloads that exceed LangSmith's 20MB limit.
    """
    if _depth > 20:
        return repr(data)[:max_str_len]
    if isinstance(data, bytes):
        return f"<binary: {len(data):,} bytes>"
    if isinstance(data, str):
        if len(data) > max_str_len:
            return data[:max_str_len] + f"\n\n[TRUNCATED - {len(data):,} chars total]"
        return data
    if isinstance(data, dict):
        changed = False
        result = {}
        for k, v in data.items():
            new_v = truncate_for_trace(v, max_str_len, _depth + 1)
            if new_v is not v:
                changed = True
            result[k] = new_v
        return result if changed else data
    if isinstance(data, (list, tuple)):
        changed = False
        result = []
        for item in data:
            new_item = truncate_for_trace(item, max_str_len, _depth + 1)
            if new_item is not item:
                changed = True
            result.append(new_item)
        return type(data)(result) if changed else data
    # Don't recurse into arbitrary objects — too expensive and creates
    # dict copies of every pydantic model / dataclass in the state tree.
    return data
