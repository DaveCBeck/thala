"""Unit tests for module-based logging system."""

import logging
import tempfile
from datetime import date, timedelta
from pathlib import Path

from core.logging import (
    MODULE_TO_LOG,
    ModuleDispatchHandler,
    RunContextFormatter,
    ThirdPartyHandler,
    end_run,
    get_current_run_id,
    module_to_log_name,
    start_run,
)
from core.logging.run_manager import (
    _compute_log_name,
    _module_log_cache,
)


class TestModuleToLogName:
    """Tests for module_to_log_name() function."""

    def test_exact_match(self):
        """Module names that exactly match a prefix."""
        assert module_to_log_name("core.stores") == "stores"
        assert module_to_log_name("core.task_queue") == "task-queue"

    def test_submodule_match(self):
        """Submodules should match their parent prefix."""
        assert module_to_log_name("core.stores.elasticsearch.client") == "stores"
        assert module_to_log_name("core.stores.zotero.api") == "stores"
        assert module_to_log_name("core.task_queue.runner") == "task-queue"

    def test_workflow_match(self):
        """Workflow modules should match their specific log names."""
        assert module_to_log_name("workflows.research.academic_lit_review") == "lit-review"
        assert module_to_log_name("workflows.research.academic_lit_review.workflow") == "lit-review"
        assert module_to_log_name("workflows.enhance.supervision") == "supervision"
        assert module_to_log_name("workflows.enhance.editing") == "editing"
        assert module_to_log_name("workflows.output.evening_reads") == "evening-reads"

    def test_longest_prefix_wins(self):
        """When multiple prefixes match, the longest one wins."""
        assert module_to_log_name("workflows.enhance.supervision.nodes") == "supervision"
        assert module_to_log_name("workflows.shared.llm_utils") == "workflows-shared"

    def test_fallback_to_misc(self):
        """Unmapped modules should fall back to 'misc'."""
        assert module_to_log_name("unknown.module") == "misc"
        assert module_to_log_name("some.random.path") == "misc"
        assert module_to_log_name("__main__") == "misc"

    def test_caching(self):
        """Results should be cached for performance."""
        _module_log_cache.clear()

        result1 = module_to_log_name("core.stores.test_module")
        assert "core.stores.test_module" in _module_log_cache

        result2 = module_to_log_name("core.stores.test_module")
        assert result1 == result2


class TestComputeLogName:
    """Tests for _compute_log_name() internal function."""

    def test_all_mappings_valid(self):
        """All MODULE_TO_LOG entries should produce valid log names."""
        for prefix, log_name in MODULE_TO_LOG.items():
            result = _compute_log_name(prefix)
            assert result == log_name, f"Expected {prefix} -> {log_name}, got {result}"


class TestRunLifecycle:
    """Tests for start_run/end_run lifecycle."""

    def test_start_run_sets_id(self):
        """start_run should set the current run ID."""
        end_run()
        assert get_current_run_id() is None

        start_run("test-run-123")
        assert get_current_run_id() == "test-run-123"

        end_run()
        assert get_current_run_id() is None

    def test_end_run_clears_state(self):
        """end_run should clear run state."""
        start_run("test-run")
        end_run()
        assert get_current_run_id() is None

    def test_multiple_start_runs(self):
        """Starting a new run should replace the previous one."""
        start_run("run-1")
        assert get_current_run_id() == "run-1"

        start_run("run-2")
        assert get_current_run_id() == "run-2"

        end_run()


def _make_record(name: str, msg: str) -> logging.LogRecord:
    """Helper to create a LogRecord."""
    return logging.LogRecord(
        name=name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestModuleDispatchHandler:
    """Tests for ModuleDispatchHandler."""

    def test_creates_log_files(self):
        """Handler should create dated log files on emit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ModuleDispatchHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            handler.emit(_make_record("core.stores.test", "Test message"))
            handler.close()

            today = date.today().isoformat()
            log_file = log_dir / f"stores.{today}.log"
            assert log_file.exists()
            assert "Test message" in log_file.read_text()

    def test_routes_to_correct_file(self):
        """Handler should route logs to the correct module file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ModuleDispatchHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            handler.emit(_make_record("core.stores.elastic", "Store message"))
            handler.emit(_make_record("core.task_queue.runner", "Queue message"))
            handler.close()

            today = date.today().isoformat()
            assert "Store message" in (log_dir / f"stores.{today}.log").read_text()
            assert "Queue message" in (log_dir / f"task-queue.{today}.log").read_text()

    def test_parallel_tasks_append_to_same_file(self):
        """Two start_run/end_run cycles should append to the same dated file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ModuleDispatchHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            start_run("run-1")
            handler.emit(_make_record("core.stores", "Run 1 message"))
            end_run()

            start_run("run-2")
            handler.emit(_make_record("core.stores", "Run 2 message"))
            end_run()

            handler.close()

            today = date.today().isoformat()
            log_file = log_dir / f"stores.{today}.log"
            content = log_file.read_text()
            assert "Run 1 message" in content
            assert "Run 2 message" in content

    def test_close_flushes_all_files(self):
        """close() should flush and close all cached file handles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ModuleDispatchHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            for module in ["core.stores", "core.task_queue", "core.scraping"]:
                handler.emit(_make_record(module, f"Message from {module}"))

            handler.close()
            assert len(handler._file_cache) == 0

    def test_cleanup_removes_old_files(self):
        """Cleanup should remove files older than retention days."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # Create an old dated log file (10 days ago)
            old_date = (date.today() - timedelta(days=10)).isoformat()
            old_file = log_dir / f"stores.{old_date}.log"
            old_file.write_text("old data")

            handler = ModuleDispatchHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            # Emit triggers cleanup
            handler.emit(_make_record("core.stores", "New message"))
            handler.close()

            assert not old_file.exists()

    def test_cleanup_preserves_recent_files(self):
        """Cleanup should keep files within retention period."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # Create a recent dated log file (2 days ago)
            recent_date = (date.today() - timedelta(days=2)).isoformat()
            recent_file = log_dir / f"stores.{recent_date}.log"
            recent_file.write_text("recent data")

            handler = ModuleDispatchHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            handler.emit(_make_record("core.stores", "New message"))
            handler.close()

            assert recent_file.exists()
            assert "recent data" in recent_file.read_text()


class TestThirdPartyHandler:
    """Tests for ThirdPartyHandler."""

    def test_writes_to_single_file(self):
        """All third-party logs should go to run-3p.{date}.log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ThirdPartyHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            for lib in ["httpx", "elasticsearch", "chromadb"]:
                handler.emit(_make_record(lib, f"Message from {lib}"))

            handler.close()

            today = date.today().isoformat()
            log_file = log_dir / f"run-3p.{today}.log"
            assert log_file.exists()
            content = log_file.read_text()
            assert "httpx" in content
            assert "elasticsearch" in content
            assert "chromadb" in content


class TestRunContextFormatter:
    """Tests for RunContextFormatter."""

    def test_with_run_id(self):
        """Formatter should embed [run_id[:8]] when a run is active."""
        fmt = RunContextFormatter("%(name)s - %(levelname)s - %(message)s")
        start_run("a94e6928-full-uuid-here")
        try:
            record = _make_record("core.stores", "Indexing document")
            result = fmt.format(record)
            assert "[a94e6928]" in result
            assert "Indexing document" in result
        finally:
            end_run()

    def test_without_run_id(self):
        """Formatter should not insert prefix when no run is active."""
        end_run()
        fmt = RunContextFormatter("%(name)s - %(levelname)s - %(message)s")
        record = _make_record("core.stores", "Indexing document")
        result = fmt.format(record)
        assert "[" not in result
        assert "Indexing document" in result

    def test_short_run_id(self):
        """Formatter should handle run IDs shorter than 8 chars."""
        fmt = RunContextFormatter("%(name)s - %(levelname)s - %(message)s")
        start_run("abc")
        try:
            record = _make_record("core.stores", "Test")
            result = fmt.format(record)
            assert "[abc]" in result
        finally:
            end_run()

    def test_integration_with_handler(self):
        """RunContextFormatter should work correctly with ModuleDispatchHandler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ModuleDispatchHandler(log_dir)
            handler.setFormatter(
                RunContextFormatter("%(name)s - %(levelname)s - %(message)s")
            )

            start_run("task-abc123")
            handler.emit(_make_record("core.stores", "Hello"))
            end_run()

            handler.close()

            today = date.today().isoformat()
            content = (log_dir / f"stores.{today}.log").read_text()
            assert "[task-abc" in content
            assert "Hello" in content
