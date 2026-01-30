---
title: Checkpoint phase_outputs JSON Serialization and Silent Exception Swallowing
date: 2026-01-30
category: storage-issues
problem_type: [storage_issue, async_issue]
module: core/task_queue/checkpoint
severity: high
symptoms:
  - "phase_outputs empty {} despite checkpoint_callback being called with outputs"
  - "Object of type datetime is not JSON serializable"
  - "Workflow resume failures - lit_result is None"
  - "Checkpoint writes fail silently with no error logs"
tags:
  - checkpoint
  - phase_outputs
  - JSON-serialization
  - async-exceptions
  - workflow-resumption
  - silent-failures
  - datetime
  - bytes
  - Path
related_docs:
  - docs/solutions/workflow-reliability/task-queue-interruption-recovery.md
  - docs/patterns/data-pipeline/incremental-checkpointing-iterative-workflows.md
  - docs/patterns/data-pipeline/task-queue-budget-tracking.md
---

# Checkpoint phase_outputs JSON Serialization and Silent Exception Swallowing

## Problem

When watching `current_work.json` during a running workflow:
- `phase` updates correctly (e.g., "lit_review" → "supervision")
- `phase_outputs` stays empty `{}` even after phases complete with outputs

This causes workflow resume failures because phase outputs like `lit_result` are `None` when loaded from checkpoint.

## Symptoms

1. **Visible in logs (after fix):**
   ```
   Checkpoint flush write 1 failed: Object of type datetime is not JSON serializable
   ```

2. **Before fix (silent):**
   - No error messages
   - `current_work.json` shows `phase_outputs: {}`
   - Resume attempts fail with `TypeError: 'NoneType' object is not subscriptable`

## Root Causes

Four issues combined to create this problem:

### Issue 1: Silent Exception Swallowing

`asyncio.gather(*tasks, return_exceptions=True)` was used without inspecting results:

```python
# BEFORE: Exceptions silently swallowed
async def await_pending_checkpoints():
    if pending_checkpoint_tasks:
        await asyncio.gather(*pending_checkpoint_tasks, return_exceptions=True)
        pending_checkpoint_tasks.clear()
        # Exceptions stored in results but never logged!
```

### Issue 2: No Task Validation

Checkpoint updates didn't validate the task existed:

```python
# BEFORE: Silent failure if task not found
for checkpoint in work["active_tasks"]:
    if checkpoint.get("task_id") == task_id:
        checkpoint["phase"] = phase
        break
# Loop completes without error even if task never found
self.storage._write_current_work_sync(work)  # Writes unchanged data
```

### Issue 3: JSON Serialization Failures

Workflow phase outputs contained non-JSON-serializable types:

| Type | Source | Example Field |
|------|--------|---------------|
| `datetime` | `PaperMetadata.retrieved_at` | `lit_result["paper_corpus"][doi]["retrieved_at"]` |
| `datetime` | `CitationEdge.discovered_at` | `lit_result["citation_edges"][*]["discovered_at"]` |
| `bytes` | `ImageOutput.image_bytes` | `series_result["image_outputs"][*]["image_bytes"]` |
| `Path` | Output paths | `illustrated_paths["lit_review"]` |

### Issue 4: Variable Shadowing

```python
# BEFORE: 'task' shadows outer parameter
def checkpoint_callback(phase, phase_outputs=None):
    async def _update():
        await checkpoint_mgr.update_checkpoint(...)

    task = asyncio.create_task(_update())  # Shadows outer 'task: Task' parameter
```

## Solution

### Fix 1: Log Exceptions from asyncio.gather

**File:** `core/task_queue/workflow_executor.py`

```python
async def await_pending_checkpoints():
    """Await all pending checkpoint tasks before cleanup/completion."""
    if pending_checkpoint_tasks:
        results = await asyncio.gather(*pending_checkpoint_tasks, return_exceptions=True)
        # Log any exceptions that occurred (don't silently swallow)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Checkpoint write {i} failed: {result}")
        pending_checkpoint_tasks.clear()
```

### Fix 2: Validate Task Exists

**File:** `core/task_queue/checkpoint/state_manager.py`

```python
def _update_checkpoint_sync(self, task_id, phase, phase_outputs=None, **kwargs):
    with self._write_lock:
        work = self.storage._read_current_work_sync()

        task_found = False
        for checkpoint in work["active_tasks"]:
            cp_task_id = checkpoint.get("task_id") or checkpoint.get("topic_id")
            if cp_task_id == task_id:
                task_found = True
                checkpoint["phase"] = phase
                # ... update logic ...
                break

        if not task_found:
            logger.error(
                f"Checkpoint update failed: task {task_id} not found in active_tasks. "
                f"Available tasks: {[c.get('task_id') for c in work['active_tasks']]}"
            )
            raise ValueError(f"Task {task_id} not found in active_tasks")

        self.storage._write_current_work_sync(work)
```

### Fix 3: Custom JSON Encoder

**File:** `core/task_queue/checkpoint/storage.py`

```python
from datetime import date, datetime
from pathlib import Path

class CheckpointJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable objects in checkpoint data."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, bytes):
            # Skip bytes (e.g., image data) - too large for checkpoints
            return f"<bytes: {len(obj)} bytes skipped>"
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def _write_current_work_sync(self, work):
    temp_file = self.current_work_file.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(temp_file, "w") as f:
            json.dump(work, f, indent=2, cls=CheckpointJSONEncoder)
        temp_file.rename(self.current_work_file)
    except Exception:
        temp_file.unlink(missing_ok=True)
        raise
```

### Fix 4: Rename Shadowed Variable

**File:** `core/task_queue/workflow_executor.py`

```python
# Renamed to avoid shadowing outer 'task' parameter
checkpoint_task = asyncio.create_task(_update())
pending_checkpoint_tasks.append(checkpoint_task)
```

## Prevention

### Best Practices

1. **Never use `return_exceptions=True` without inspecting results**
   - Always loop through results and log/handle exceptions

2. **Validate state before operations**
   - Check task exists before updating
   - Raise explicit errors on invalid state

3. **Use custom JSON encoders for complex data**
   - Handle datetime, date, bytes, Path at minimum
   - Consider what data is actually needed for resume vs. what can be skipped

4. **Avoid variable shadowing in callbacks**
   - Use distinct names for asyncio.Task variables

### Test Cases

```python
async def test_phase_outputs_persistence_across_phases():
    """Verify phase_outputs persist when transitioning between phases."""
    manager = CheckpointManager(queue_dir=Path(tmpdir))

    await manager.start_work(task_id, "lit_review_full", "run-123")
    await manager.update_checkpoint(task_id, "lit_review",
                                    phase_outputs={"lit_result": {"report": "test"}})
    await manager.update_checkpoint(task_id, "supervision")  # No outputs

    checkpoint = await manager.get_checkpoint(task_id)
    assert "lit_result" in checkpoint.get("phase_outputs", {}), \
        "lit_result should persist after phase transition"

async def test_checkpoint_update_fails_for_missing_task():
    """Verify checkpoint update raises error if task not found."""
    manager = CheckpointManager(queue_dir=Path(tmpdir))

    with pytest.raises(ValueError, match="not found in active_tasks"):
        await manager.update_checkpoint("nonexistent-task", "phase")

async def test_checkpoint_with_complex_types():
    """Verify datetime, Path, and bytes are serializable."""
    manager = CheckpointManager(queue_dir=Path(tmpdir))
    await manager.start_work(task_id, "lit_review_full", "run-123")

    await manager.update_checkpoint(task_id, "research", phase_outputs={
        "timestamp": datetime.now(),
        "output_path": Path("/some/path"),
        "image_data": b"fake bytes",
    })

    checkpoint = await manager.get_checkpoint(task_id)
    assert checkpoint is not None  # No serialization error
```

## Files Modified

| File | Change |
|------|--------|
| `core/task_queue/workflow_executor.py` | Log exceptions, fix variable shadowing |
| `core/task_queue/checkpoint/state_manager.py` | Validate task exists, add diagnostic logging |
| `core/task_queue/checkpoint/storage.py` | Add `CheckpointJSONEncoder` |
| `testing/test_task_queue_interruption.py` | Add regression tests |

## Verification

1. Run tests: `PYTHONPATH=/home/dave/thala python testing/test_task_queue_interruption.py`
2. Run workflow and watch for log: `"Checkpoint {task_id}: saving phase_outputs keys = ['lit_result']"`
3. Check `current_work.json` shows populated `phase_outputs`
4. CTRL-C during supervision, resume, verify `lit_result` loads correctly

## Related Documentation

- [Task Queue Interruption Recovery](../workflow-reliability/task-queue-interruption-recovery.md)
- [Incremental Checkpointing Pattern](../../patterns/data-pipeline/incremental-checkpointing-iterative-workflows.md)
- [Task Queue Budget Tracking](../../patterns/data-pipeline/task-queue-budget-tracking.md)
