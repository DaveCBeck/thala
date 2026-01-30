"""Unit tests for module-based logging system."""

import logging
import tempfile
from pathlib import Path

from core.logging import (
    MODULE_TO_LOG,
    ModuleDispatchHandler,
    ThirdPartyHandler,
    end_run,
    get_current_run_id,
    module_to_log_name,
    start_run,
)
from core.logging.run_manager import (
    _compute_log_name,
    _module_log_cache,
    should_rotate,
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
        # "workflows.enhance.supervision" is longer than "workflows.enhance"
        # so it should win
        assert module_to_log_name("workflows.enhance.supervision.nodes") == "supervision"
        # "workflows.shared" is a specific mapping
        assert module_to_log_name("workflows.shared.llm_utils") == "workflows-shared"

    def test_fallback_to_misc(self):
        """Unmapped modules should fall back to 'misc'."""
        assert module_to_log_name("unknown.module") == "misc"
        assert module_to_log_name("some.random.path") == "misc"
        assert module_to_log_name("__main__") == "misc"

    def test_caching(self):
        """Results should be cached for performance."""
        # Clear cache first
        _module_log_cache.clear()

        # First call computes
        result1 = module_to_log_name("core.stores.test_module")
        assert "core.stores.test_module" in _module_log_cache

        # Second call uses cache
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
        end_run()  # Clean state
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


class TestShouldRotate:
    """Tests for should_rotate() function."""

    def test_no_rotation_without_run(self):
        """should_rotate returns False when no run is active."""
        end_run()  # Clean state
        assert should_rotate("test-log") is False

    def test_rotation_on_first_access(self):
        """should_rotate returns True on first access within a run."""
        start_run("test-run")
        assert should_rotate("stores") is True
        end_run()

    def test_no_double_rotation(self):
        """should_rotate returns False for same log file within same run."""
        start_run("test-run")
        assert should_rotate("stores") is True
        assert should_rotate("stores") is False  # Already rotated
        end_run()

    def test_different_logs_rotate_independently(self):
        """Different log files rotate independently within a run."""
        start_run("test-run")
        assert should_rotate("stores") is True
        assert should_rotate("task-queue") is True
        assert should_rotate("stores") is False
        assert should_rotate("task-queue") is False
        end_run()

    def test_new_run_resets_rotation(self):
        """Starting a new run should allow rotation again."""
        start_run("run-1")
        assert should_rotate("stores") is True
        end_run()

        start_run("run-2")
        assert should_rotate("stores") is True  # Should rotate again in new run
        end_run()


class TestModuleDispatchHandler:
    """Tests for ModuleDispatchHandler."""

    def test_creates_log_files(self):
        """Handler should create log files on emit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ModuleDispatchHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            # Create a log record
            record = logging.LogRecord(
                name="core.stores.test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Test message",
                args=(),
                exc_info=None,
            )

            handler.emit(record)
            handler.close()

            # Check file was created
            log_file = log_dir / "stores.log"
            assert log_file.exists()
            assert "Test message" in log_file.read_text()

    def test_routes_to_correct_file(self):
        """Handler should route logs to the correct module file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ModuleDispatchHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            # Log from two different modules
            for module_name, message in [
                ("core.stores.elastic", "Store message"),
                ("core.task_queue.runner", "Queue message"),
            ]:
                record = logging.LogRecord(
                    name=module_name,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=message,
                    args=(),
                    exc_info=None,
                )
                handler.emit(record)

            handler.close()

            # Check files
            assert "Store message" in (log_dir / "stores.log").read_text()
            assert "Queue message" in (log_dir / "task-queue.log").read_text()

    def test_rotation_on_new_run(self):
        """Handler should rotate files when a new run starts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ModuleDispatchHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            # First run
            start_run("run-1")
            record1 = logging.LogRecord(
                name="core.stores",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Run 1 message",
                args=(),
                exc_info=None,
            )
            handler.emit(record1)
            end_run()

            # Second run
            start_run("run-2")
            record2 = logging.LogRecord(
                name="core.stores",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Run 2 message",
                args=(),
                exc_info=None,
            )
            handler.emit(record2)
            end_run()

            handler.close()

            # Current file should have run 2 message only
            current = log_dir / "stores.log"
            previous = log_dir / "stores.previous.log"

            assert current.exists()
            assert previous.exists()
            assert "Run 2 message" in current.read_text()
            assert "Run 1 message" in previous.read_text()

    def test_close_flushes_all_files(self):
        """close() should flush and close all cached file handles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ModuleDispatchHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            # Emit to multiple modules
            for module in ["core.stores", "core.task_queue", "core.scraping"]:
                record = logging.LogRecord(
                    name=module,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=f"Message from {module}",
                    args=(),
                    exc_info=None,
                )
                handler.emit(record)

            # Close should clean up
            handler.close()
            assert len(handler._file_cache) == 0


class TestThirdPartyHandler:
    """Tests for ThirdPartyHandler."""

    def test_writes_to_single_file(self):
        """All third-party logs should go to run-3p.log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ThirdPartyHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            # Simulate logs from different third-party libraries
            for lib in ["httpx", "elasticsearch", "chromadb"]:
                record = logging.LogRecord(
                    name=lib,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=f"Message from {lib}",
                    args=(),
                    exc_info=None,
                )
                handler.emit(record)

            handler.close()

            # All messages should be in single file
            log_file = log_dir / "run-3p.log"
            assert log_file.exists()
            content = log_file.read_text()
            assert "httpx" in content
            assert "elasticsearch" in content
            assert "chromadb" in content

    def test_rotation_on_new_run(self):
        """Third-party handler should rotate on new run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            handler = ThirdPartyHandler(log_dir)
            handler.setFormatter(logging.Formatter("%(message)s"))

            # First run
            start_run("run-1")
            record1 = logging.LogRecord(
                name="httpx",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Run 1 httpx message",
                args=(),
                exc_info=None,
            )
            handler.emit(record1)
            end_run()

            # Second run
            start_run("run-2")
            record2 = logging.LogRecord(
                name="httpx",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Run 2 httpx message",
                args=(),
                exc_info=None,
            )
            handler.emit(record2)
            end_run()

            handler.close()

            # Check rotation
            current = log_dir / "run-3p.log"
            previous = log_dir / "run-3p.previous.log"

            assert current.exists()
            assert previous.exists()
            assert "Run 2" in current.read_text()
            assert "Run 1" in previous.read_text()
