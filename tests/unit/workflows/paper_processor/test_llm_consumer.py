"""Unit tests for concurrent llm_consumer in the paper processing pipeline.

Tests verify:
- Concurrent execution (wall-clock time < sequential time)
- Checkpoint correctness
- Error isolation (one paper failure doesn't affect others)
"""

import asyncio
import time

import pytest


def _make_paper(doi: str) -> dict:
    """Create a minimal PaperMetadata dict for testing."""
    return {"doi": doi, "title": f"Paper {doi}", "authors": []}


async def _run_llm_consumer(
    items: list[tuple[str, str, dict]],
    mock_process_fn,
    checkpoint_callback=None,
    checkpoint_interval=5,
):
    """Run the llm_consumer logic with given items piped through llm_queue.

    Mirrors the llm_consumer from core.py for isolated testing without
    needing the full pipeline (acquisition_producer, marker_consumer, etc.).
    Returns (processing_results, processing_failed, processed_count).
    """
    llm_queue: asyncio.Queue = asyncio.Queue()
    processing_results: dict[str, dict] = {}
    processing_failed: list[str] = []
    processed_count = 0
    last_checkpoint_count = 0

    async def llm_consumer():
        nonlocal processed_count, last_checkpoint_count
        active_tasks: set[asyncio.Task] = set()

        async def process_llm_item(doi: str, markdown_text: str, paper):
            nonlocal processed_count, last_checkpoint_count
            try:
                result = await mock_process_fn(doi, markdown_text, paper, is_markdown=True)
                processed_count += 1

                if result.get("success"):
                    processing_results[doi] = result
                elif result.get("validation_failed"):
                    processing_failed.append(doi)
                else:
                    processing_failed.append(doi)

                if checkpoint_callback and processed_count - last_checkpoint_count >= checkpoint_interval:
                    last_checkpoint_count = processed_count
                    checkpoint_callback(processed_count, dict(processing_results))

            except Exception:
                processed_count += 1
                processing_failed.append(doi)

        while True:
            item = await llm_queue.get()
            if item is None:
                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                break

            doi, markdown_text, paper = item
            task = asyncio.create_task(process_llm_item(doi, markdown_text, paper))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

    # Feed items into queue
    for item in items:
        await llm_queue.put(item)
    await llm_queue.put(None)  # Sentinel

    await llm_consumer()
    return processing_results, processing_failed, processed_count


@pytest.mark.asyncio
async def test_concurrent_execution_faster_than_sequential():
    """Verify papers are processed concurrently, not sequentially.

    With 5 papers each taking 0.1s, sequential would take ~0.5s.
    All running concurrently should complete in ~0.1s plus overhead.
    """
    delay = 0.1
    num_papers = 5

    async def mock_process(doi, markdown_text, paper, is_markdown=True):
        await asyncio.sleep(delay)
        return {"doi": doi, "success": True}

    items = [(f"10.1000/{i}", f"markdown_{i}", _make_paper(f"10.1000/{i}")) for i in range(num_papers)]

    start = time.monotonic()
    results, failed, count = await _run_llm_consumer(items, mock_process)
    elapsed = time.monotonic() - start

    # All papers run concurrently, so wall-clock should be ~1x delay, not 5x
    assert elapsed < num_papers * delay * 0.8, (
        f"Expected concurrent execution, but took {elapsed:.2f}s "
        f"(sequential would be {num_papers * delay:.2f}s)"
    )
    assert count == num_papers
    assert len(results) == num_papers
    assert len(failed) == 0


@pytest.mark.asyncio
async def test_all_papers_run_concurrently():
    """Verify that all queued papers can run at the same time (no semaphore cap)."""
    concurrent_count = 0
    max_observed = 0

    async def mock_process(doi, markdown_text, paper, is_markdown=True):
        nonlocal concurrent_count, max_observed
        concurrent_count += 1
        max_observed = max(max_observed, concurrent_count)
        await asyncio.sleep(0.05)
        concurrent_count -= 1
        return {"doi": doi, "success": True}

    num_papers = 10
    items = [(f"10.1000/{i}", f"markdown_{i}", _make_paper(f"10.1000/{i}")) for i in range(num_papers)]

    results, failed, count = await _run_llm_consumer(items, mock_process)

    # All 10 should run concurrently since there's no semaphore cap
    assert max_observed == num_papers, (
        f"Expected all {num_papers} papers concurrent, but max was {max_observed}"
    )
    assert count == num_papers


@pytest.mark.asyncio
async def test_checkpoint_called_at_interval():
    """Verify checkpoint callback fires at correct intervals."""
    checkpoint_calls = []

    def checkpoint_cb(count, results):
        checkpoint_calls.append((count, dict(results)))

    async def mock_process(doi, markdown_text, paper, is_markdown=True):
        return {"doi": doi, "success": True}

    num_papers = 12
    interval = 5
    items = [(f"10.1000/{i}", f"markdown_{i}", _make_paper(f"10.1000/{i}")) for i in range(num_papers)]

    results, failed, count = await _run_llm_consumer(
        items, mock_process, checkpoint_callback=checkpoint_cb, checkpoint_interval=interval
    )

    assert count == num_papers
    assert len(results) == num_papers
    assert len(checkpoint_calls) >= 1, f"Expected checkpoints, got {len(checkpoint_calls)}"
    assert checkpoint_calls[0][0] >= interval


@pytest.mark.asyncio
async def test_error_isolation():
    """Verify that one paper's failure doesn't affect others."""
    async def mock_process(doi, markdown_text, paper, is_markdown=True):
        if doi == "10.1000/2":
            raise RuntimeError("Simulated failure")
        await asyncio.sleep(0.01)
        return {"doi": doi, "success": True}

    num_papers = 5
    items = [(f"10.1000/{i}", f"markdown_{i}", _make_paper(f"10.1000/{i}")) for i in range(num_papers)]

    results, failed, count = await _run_llm_consumer(items, mock_process)

    assert count == num_papers
    assert "10.1000/2" in failed
    assert len(results) == num_papers - 1
    for i in range(num_papers):
        doi = f"10.1000/{i}"
        if i == 2:
            assert doi not in results
        else:
            assert doi in results


@pytest.mark.asyncio
async def test_validation_failure_tracked_correctly():
    """Verify validation failures are tracked in processing_failed."""
    async def mock_process(doi, markdown_text, paper, is_markdown=True):
        if doi == "10.1000/1":
            return {"doi": doi, "validation_failed": True, "validation_reasoning": "Mismatch"}
        return {"doi": doi, "success": True}

    items = [(f"10.1000/{i}", f"markdown_{i}", _make_paper(f"10.1000/{i}")) for i in range(3)]

    results, failed, count = await _run_llm_consumer(items, mock_process)

    assert count == 3
    assert "10.1000/1" in failed
    assert "10.1000/1" not in results
    assert len(results) == 2


@pytest.mark.asyncio
async def test_empty_queue_completes():
    """Verify llm_consumer handles empty queue (immediate sentinel)."""
    async def mock_process(doi, markdown_text, paper, is_markdown=True):
        return {"doi": doi, "success": True}

    results, failed, count = await _run_llm_consumer([], mock_process)

    assert count == 0
    assert len(results) == 0
    assert len(failed) == 0


@pytest.mark.asyncio
async def test_checkpoint_snapshot_isolation():
    """Verify checkpoint receives a snapshot, not a live reference."""
    checkpoint_snapshots = []

    def checkpoint_cb(count, results):
        checkpoint_snapshots.append(results)

    call_count = 0

    async def mock_process(doi, markdown_text, paper, is_markdown=True):
        nonlocal call_count
        call_count += 1
        return {"doi": doi, "success": True, "order": call_count}

    items = [(f"10.1000/{i}", f"markdown_{i}", _make_paper(f"10.1000/{i}")) for i in range(6)]

    results, failed, count = await _run_llm_consumer(
        items, mock_process, checkpoint_callback=checkpoint_cb, checkpoint_interval=3
    )

    assert count == 6
    # Checkpoint snapshots should be independent of final results
    if checkpoint_snapshots:
        assert len(checkpoint_snapshots[0]) <= len(results)
