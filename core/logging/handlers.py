"""Custom logging handlers for module-based dispatch.

Provides ModuleDispatchHandler which routes log records to per-module files
based on the logger name, using the MODULE_TO_LOG mapping.

Note on Async Contexts:
    These handlers perform synchronous file I/O (write, flush, unlink, rename).
    While this technically blocks the event loop when used from async code, the
    impact is typically 10-50 microseconds per operation, which is acceptable
    for most workloads. This is by design: Python's standard logging module is
    synchronous, and making it async would add significant complexity.

    For high-throughput async applications where even microsecond-level blocking
    is unacceptable, consider using logging.handlers.QueueHandler to offload
    I/O to a background thread.
"""

import logging
from pathlib import Path
from typing import TextIO


def _rotate_log_file(log_dir: Path, log_name: str, stream: TextIO | None) -> TextIO:
    """Rotate log file and return new file handle.

    Renames current.log -> previous.log, deleting old previous.log if exists.
    Closes the existing stream if provided.

    Args:
        log_dir: Directory containing log files
        log_name: Base name of the log file (without .log extension)
        stream: Existing stream to close, or None

    Returns:
        New opened file handle for writing.
    """
    current = log_dir / f"{log_name}.log"
    previous = log_dir / f"{log_name}.previous.log"

    # Close existing stream if open
    if stream:
        stream.close()

    # Rotate files
    if previous.exists():
        previous.unlink()
    if current.exists():
        current.rename(previous)

    return current.open("a", encoding="utf-8")


class ModuleDispatchHandler(logging.Handler):
    """Single handler that routes records to per-module log files.

    Avoids file handle exhaustion by using an internal cache instead of
    creating separate FileHandler instances per module. On Linux, the default
    ulimit is ~1024, so creating many FileHandlers can exhaust available handles.

    Features:
    - Lazy file opening (files created on first write)
    - Run-based rotation via start_run() (current.log → previous.log)
    - Max 2 files per module (current + previous)
    - Thread-safe through Python's logging lock mechanism

    Usage:
        handler = ModuleDispatchHandler(Path("logs"))
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logging.getLogger().addHandler(handler)
    """

    def __init__(self, log_dir: Path):
        """Initialize the handler.

        Args:
            log_dir: Directory where log files will be created
        """
        super().__init__()
        self.log_dir = log_dir
        self._file_cache: dict[str, TextIO] = {}

    def emit(self, record: logging.LogRecord) -> None:
        """Route a log record to the appropriate module file.

        Args:
            record: The log record to emit
        """
        try:
            # Import here to avoid circular imports
            from core.logging.run_manager import module_to_log_name, should_rotate

            log_name = module_to_log_name(record.name)

            # Check if rotation is needed (first log to this module in current run)
            if should_rotate(log_name):
                self._rotate_file(log_name)

            file = self._get_or_open_file(log_name)
            msg = self.format(record)
            file.write(msg + "\n")
            file.flush()

        except Exception:
            self.handleError(record)

    def _rotate_file(self, log_name: str) -> None:
        """Rotate current.log -> previous.log, delete old previous.

        Args:
            log_name: The log file name (without extension)
        """
        # Get existing stream from cache (may be None)
        existing_stream = self._file_cache.pop(log_name, None)

        # Use shared rotation logic - returns new file handle
        new_stream = _rotate_log_file(self.log_dir, log_name, existing_stream)

        # Cache the new stream
        self._file_cache[log_name] = new_stream

    def _get_or_open_file(self, log_name: str) -> TextIO:
        """Get or open file handle for log_name.

        Args:
            log_name: The log file name (without extension)

        Returns:
            Open file handle for writing
        """
        if log_name not in self._file_cache:
            path = self.log_dir / f"{log_name}.log"
            self._file_cache[log_name] = open(path, "a", encoding="utf-8")
        return self._file_cache[log_name]

    def close(self) -> None:
        """Close all cached file handles.

        Called when the handler is removed from a logger or at shutdown.
        """
        self.acquire()
        try:
            for file in self._file_cache.values():
                try:
                    file.close()
                except Exception:
                    pass  # Best effort
            self._file_cache.clear()
        finally:
            self.release()
        super().close()


class ThirdPartyHandler(logging.FileHandler):
    """Handler for third-party library logs.

    All third-party logs go to a single run-3p.log file.
    Supports run-based rotation like ModuleDispatchHandler.
    """

    LOG_NAME = "run-3p"

    def __init__(self, log_dir: Path, **kwargs):
        """Initialize the handler.

        Args:
            log_dir: Directory where log file will be created
            **kwargs: Additional arguments passed to FileHandler
        """
        self.log_dir = log_dir
        log_file = log_dir / f"{self.LOG_NAME}.log"
        super().__init__(log_file, mode="a", encoding="utf-8", **kwargs)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record, rotating if needed.

        Args:
            record: The log record to emit
        """
        try:
            from core.logging.run_manager import should_rotate

            if should_rotate(self.LOG_NAME):
                self._rotate_file()

            super().emit(record)

        except Exception:
            self.handleError(record)

    def _rotate_file(self) -> None:
        """Rotate current run-3p.log -> run-3p.previous.log."""
        # Use shared rotation logic - returns new file handle
        self.stream = _rotate_log_file(self.log_dir, self.LOG_NAME, self.stream)
