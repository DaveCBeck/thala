"""Custom logging handlers for module-based dispatch with date-stamped files.

Provides ModuleDispatchHandler which routes log records to per-module files
based on the logger name, using the MODULE_TO_LOG mapping. Files are named
{module}.{YYYY-MM-DD}.log and appended to throughout the day.

Also provides RunContextFormatter which embeds [run_id] in each log line so
parallel tasks are distinguishable via grep.

Note on Async Contexts:
    These handlers perform synchronous file I/O (write, flush).
    While this technically blocks the event loop when used from async code, the
    impact is typically 10-50 microseconds per operation, which is acceptable
    for most workloads. This is by design: Python's standard logging module is
    synchronous, and making it async would add significant complexity.
"""

import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TextIO

# Matches files like "stores.2026-02-10.log" or "run-3p.2025-12-31.log"
_DATED_LOG_RE = re.compile(r"^.+\.(\d{4}-\d{2}-\d{2})\.log$")

_RETENTION_DAYS_DEFAULT = 7


def _cleanup_old_logs(log_dir: Path, retention_days: int) -> None:
    """Delete dated log files older than retention_days.

    Scans for files matching *.YYYY-MM-DD.log and removes those with dates
    older than the cutoff.

    Args:
        log_dir: Directory containing log files
        retention_days: Number of days to retain
    """
    cutoff = date.today() - timedelta(days=retention_days)
    for path in log_dir.glob("*.*.log"):
        match = _DATED_LOG_RE.match(path.name)
        if match:
            try:
                file_date = date.fromisoformat(match.group(1))
                if file_date < cutoff:
                    path.unlink()
            except ValueError:
                pass  # Not a valid date, skip


class RunContextFormatter(logging.Formatter):
    """Formatter that inserts [run_id] before the message portion.

    When a run is active (via start_run()), log lines look like:
        2026-02-10 18:03:27 - core.stores - INFO - [a94e6928] Indexing document

    When no run is active, log lines are unchanged:
        2026-02-10 18:03:27 - core.stores - INFO - Indexing document
    """

    def format(self, record: logging.LogRecord) -> str:
        from core.logging.run_manager import get_current_run_id

        result = super().format(record)

        run_id = get_current_run_id()
        if run_id:
            prefix = f"[{run_id[:8]}] "
            # Insert after the last " - " separator (before the message)
            last_sep = result.rfind(" - ")
            if last_sep != -1:
                insert_pos = last_sep + 3
                result = result[:insert_pos] + prefix + result[insert_pos:]

        return result


class ModuleDispatchHandler(logging.Handler):
    """Single handler that routes records to per-module date-stamped log files.

    Files are named {module}.{YYYY-MM-DD}.log and appended to throughout the
    day. When the date changes, stale handles are closed and new ones opened.

    On first log of each new day, old log files beyond the retention period
    are cleaned up.

    Features:
    - Lazy file opening (files created on first write)
    - Date-based files (safe for parallel workflows)
    - Automatic cleanup of old dated files
    - Thread-safe through Python's logging lock mechanism

    Usage:
        handler = ModuleDispatchHandler(Path("logs"))
        handler.setFormatter(RunContextFormatter("%(asctime)s - %(message)s"))
        logging.getLogger().addHandler(handler)
    """

    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir
        self._file_cache: dict[tuple[str, str], TextIO] = {}
        self._last_cleanup_date: str | None = None
        self._retention_days = int(
            os.getenv("THALA_LOG_RETENTION_DAYS", str(_RETENTION_DAYS_DEFAULT))
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from core.logging.run_manager import module_to_log_name

            log_name = module_to_log_name(record.name)
            today = date.today().isoformat()

            self._maybe_cleanup(today)

            file = self._get_or_open_file(log_name, today)
            msg = self.format(record)
            file.write(msg + "\n")
            file.flush()

        except Exception:
            self.handleError(record)

    def _maybe_cleanup(self, today: str) -> None:
        """Run cleanup once per day on first log of a new date."""
        if self._last_cleanup_date != today:
            # Close stale handles from previous dates
            stale_keys = [
                key for key in self._file_cache if key[1] != today
            ]
            for key in stale_keys:
                try:
                    self._file_cache.pop(key).close()
                except Exception:
                    pass

            _cleanup_old_logs(self.log_dir, self._retention_days)
            self._last_cleanup_date = today

    def _get_or_open_file(self, log_name: str, date_str: str) -> TextIO:
        key = (log_name, date_str)
        if key not in self._file_cache:
            path = self.log_dir / f"{log_name}.{date_str}.log"
            self._file_cache[key] = open(path, "a", encoding="utf-8")
        return self._file_cache[key]

    def close(self) -> None:
        self.acquire()
        try:
            for file in self._file_cache.values():
                try:
                    file.close()
                except Exception:
                    pass
            self._file_cache.clear()
        finally:
            self.release()
        super().close()


class ThirdPartyHandler(logging.Handler):
    """Handler for third-party library logs.

    All third-party logs go to a single run-3p.{YYYY-MM-DD}.log file.
    Uses the same date-based pattern as ModuleDispatchHandler.
    """

    LOG_NAME = "run-3p"

    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir
        self._stream: TextIO | None = None
        self._current_date: str | None = None
        self._last_cleanup_date: str | None = None
        self._retention_days = int(
            os.getenv("THALA_LOG_RETENTION_DAYS", str(_RETENTION_DAYS_DEFAULT))
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            today = date.today().isoformat()

            if self._last_cleanup_date != today:
                _cleanup_old_logs(self.log_dir, self._retention_days)
                self._last_cleanup_date = today

            if self._current_date != today:
                if self._stream:
                    self._stream.close()
                path = self.log_dir / f"{self.LOG_NAME}.{today}.log"
                self._stream = open(path, "a", encoding="utf-8")
                self._current_date = today

            msg = self.format(record)
            self._stream.write(msg + "\n")
            self._stream.flush()

        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self.acquire()
        try:
            if self._stream:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        finally:
            self.release()
        super().close()
