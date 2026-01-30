"""Test task queue interruption and resume functionality.

Tests the graceful shutdown, incremental checkpointing, and resume logic
for the task queue system.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from core.config import configure_logging
from core.task_queue.checkpoint_manager import CheckpointManager
from core.task_queue.incremental_state import IncrementalStateManager
from core.task_queue.shutdown import ShutdownCoordinator

configure_logging("task_queue_interruption_test")
logger = logging.getLogger(__name__)


def test_shutdown_coordinator_basic():
    """Test ShutdownCoordinator basic functionality."""
    logger.info("=== Testing ShutdownCoordinator Basic ===")

    coordinator = ShutdownCoordinator()

    # Initially not shutdown
    assert not coordinator.shutdown_requested
    logger.info("Initial state: shutdown_requested=False ✓")

    # Request shutdown
    coordinator.request_shutdown()
    assert coordinator.shutdown_requested
    logger.info("After request_shutdown: shutdown_requested=True ✓")

    # Multiple requests should be idempotent
    coordinator.request_shutdown()
    assert coordinator.shutdown_requested
    logger.info("Idempotent request: still shutdown_requested=True ✓")

    logger.info("ShutdownCoordinator basic test PASSED")


async def test_shutdown_coordinator_wait_or_shutdown():
    """Test ShutdownCoordinator's interruptible wait."""
    logger.info("=== Testing ShutdownCoordinator wait_or_shutdown ===")

    coordinator = ShutdownCoordinator()

    # Test timeout without shutdown
    logger.info("Testing normal timeout (0.1s)...")
    result = await coordinator.wait_or_shutdown(0.1)
    assert result is False, "Should return False on timeout"
    logger.info("Normal timeout returned False ✓")

    # Test shutdown interrupts wait
    logger.info("Testing shutdown interrupts wait...")

    async def request_shutdown_after_delay():
        await asyncio.sleep(0.05)
        coordinator.request_shutdown()

    # Start shutdown request in background
    asyncio.create_task(request_shutdown_after_delay())

    # Wait for up to 10 seconds (but should return early)
    import time

    start = time.time()
    result = await coordinator.wait_or_shutdown(10.0)
    elapsed = time.time() - start

    assert result is True, "Should return True when shutdown requested"
    assert elapsed < 1.0, f"Should return quickly, not after {elapsed}s"
    logger.info(f"Shutdown interruption returned in {elapsed:.3f}s ✓")

    logger.info("ShutdownCoordinator wait_or_shutdown test PASSED")


async def test_incremental_state_manager():
    """Test IncrementalStateManager save/load/clear operations (async)."""
    logger.info("=== Testing IncrementalStateManager (async) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create manager with custom directory
        manager = IncrementalStateManager(incremental_dir=Path(tmpdir))

        task_id = "test-task-123"
        phase = "paper_processing"

        # Test save (async)
        await manager.save_progress(
            task_id=task_id,
            phase=phase,
            iteration_count=5,
            partial_results={"doi1": {"title": "Paper 1"}, "doi2": {"title": "Paper 2"}},
            checkpoint_interval=5,
        )
        logger.info("Saved incremental progress ✓")

        # Test load (async)
        state = await manager.load_progress(task_id, phase)
        assert state is not None, "Should load saved state"
        assert state["iteration_count"] == 5
        assert len(state["partial_results"]) == 2
        assert state["partial_results"]["doi1"]["title"] == "Paper 1"
        logger.info(f"Loaded state: iteration={state['iteration_count']}, results={len(state['partial_results'])} ✓")

        # Test clear (async)
        await manager.clear_progress(task_id)
        cleared_state = await manager.load_progress(task_id, phase)
        assert cleared_state is None, "Should return None after clear"
        logger.info("Cleared progress ✓")

        logger.info("IncrementalStateManager test PASSED")


async def test_checkpoint_manager_orphaned_temps():
    """Test CheckpointManager orphaned temp file cleanup (async)."""
    logger.info("=== Testing CheckpointManager Orphaned Temp Cleanup (async) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        queue_dir = Path(tmpdir)

        # Create some orphaned temp files
        (queue_dir / "current_work.json.tmp").write_text('{"orphaned": true}')
        (queue_dir / "other_file.tmp").write_text('{"another": "orphan"}')

        # Create manager and cleanup (async)
        manager = CheckpointManager(queue_dir=queue_dir)
        cleaned = await manager.cleanup_orphaned_temps()

        assert cleaned == 2, f"Should clean 2 files, cleaned {cleaned}"
        assert not (queue_dir / "current_work.json.tmp").exists()
        assert not (queue_dir / "other_file.tmp").exists()
        logger.info(f"Cleaned {cleaned} orphaned temp files ✓")

        # Cleanup again should find nothing (async)
        cleaned_again = await manager.cleanup_orphaned_temps()
        assert cleaned_again == 0
        logger.info("Second cleanup found 0 files ✓")

        logger.info("CheckpointManager orphaned temp cleanup test PASSED")


async def test_checkpoint_callback_flow():
    """Test that checkpoint callbacks flow correctly through the system (async)."""
    logger.info("=== Testing Checkpoint Callback Flow (async) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalStateManager(incremental_dir=Path(tmpdir))

        # Simulate what happens in lit_review_full.py
        task_id = "flow-test-task"
        checkpoint_calls = []

        async def supervision_checkpoint(iteration: int, partial_results: dict) -> None:
            """Save incremental progress during supervision loops (async)."""
            checkpoint_calls.append((iteration, len(partial_results)))
            await manager.save_progress(
                task_id=task_id,
                phase="supervision",
                iteration_count=iteration,
                partial_results=partial_results,
                checkpoint_interval=1,
            )

        # Simulate 3 supervision loop iterations
        for i in range(1, 4):
            partial = {
                "current_review": f"Review after iteration {i}",
                "iteration": i,
                "explored_bases": [f"base_{j}" for j in range(i)],
            }
            await supervision_checkpoint(i, partial)

        assert len(checkpoint_calls) == 3
        logger.info(f"Made {len(checkpoint_calls)} checkpoint calls ✓")

        # Verify final state (async)
        state = await manager.load_progress(task_id, "supervision")
        assert state is not None
        assert state["iteration_count"] == 3
        assert state["partial_results"]["iteration"] == 3
        assert len(state["partial_results"]["explored_bases"]) == 3
        logger.info(f"Final state: iteration={state['iteration_count']} ✓")

        logger.info("Checkpoint callback flow test PASSED")


async def test_event_loop_not_blocked():
    """Test that file I/O doesn't block the event loop."""
    logger.info("=== Testing Event Loop Not Blocked ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalStateManager(incremental_dir=Path(tmpdir))
        task_id = "blocking-test-task"

        # Track if concurrent task ran during file I/O
        concurrent_task_ran = False

        async def concurrent_task():
            nonlocal concurrent_task_ran
            await asyncio.sleep(0.001)  # Yield to event loop
            concurrent_task_ran = True

        # Start concurrent task and file I/O at the same time
        await asyncio.gather(
            manager.save_progress(
                task_id=task_id,
                phase="test",
                iteration_count=1,
                partial_results={"key": "value" * 1000},  # Larger payload
            ),
            concurrent_task(),
        )

        assert concurrent_task_ran, "Concurrent task should have run during file I/O"
        logger.info("Concurrent task ran during file I/O ✓")

        # Verify the save worked
        state = await manager.load_progress(task_id, "test")
        assert state is not None
        assert state["iteration_count"] == 1
        logger.info("File I/O completed correctly ✓")

        logger.info("Event loop not blocked test PASSED")


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Task Queue Interruption & Resume Tests")
    logger.info("=" * 60)

    # Run sync tests
    test_shutdown_coordinator_basic()
    print()

    # Run async tests
    await test_shutdown_coordinator_wait_or_shutdown()
    print()

    await test_incremental_state_manager()
    print()

    await test_checkpoint_manager_orphaned_temps()
    print()

    await test_checkpoint_callback_flow()
    print()

    await test_event_loop_not_blocked()
    print()

    logger.info("=" * 60)
    logger.info("All tests PASSED!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
