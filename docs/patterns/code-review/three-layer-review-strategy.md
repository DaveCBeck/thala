---
title: "Three-Layer Code Review Strategy"
description: "Systematic approach to resolving code review findings across correctness, clarity, and maintainability layers"
category: pattern
subcategory: code-review
tags:
  - code-review
  - thread-safety
  - async-safety
  - api-design
  - documentation
  - logging
module: core/logging, core/task_queue, testing
difficulty: intermediate
verified: true
date_created: 2026-01-30
---

# Three-Layer Code Review Strategy

## Pattern Overview

When resolving code review findings, apply fixes systematically across three layers:

1. **Layer 1 - Correctness**: Fix bugs that manifest under edge cases or concurrency
2. **Layer 2 - Clarity**: Document constraints, trade-offs, and security properties
3. **Layer 3 - Maintainability**: Clean up APIs, eliminate duplication, improve discoverability

This pattern emerged from resolving 11 code review findings across the logging infrastructure.

## When to Apply

- Code reviews with 5+ findings across different concern areas
- Infrastructure code used in both sync and async contexts
- Modules with concurrent/multi-threaded usage patterns
- Code with implicit security properties or constraints

## The Three Layers

### Layer 1: Correctness

Fix bugs that only appear under specific conditions (concurrency, resource management, async cancellation).

**Principle**: *Assume concurrent access and state transitions will happen—protect against them proactively.*

#### Pattern 1.1: Thread-Safe Check-Then-Add

**Problem**: Non-atomic check-then-add patterns race under concurrent access.

```python
# BEFORE: Race condition
if log_name in rotated:
    return False
# Thread B could pass check here before Thread A adds
rotated.add(log_name)
return True
```

```python
# AFTER: Thread-safe with explicit lock
_rotation_lock = threading.Lock()

def should_rotate(log_name: str) -> bool:
    with _rotation_lock:
        if log_name in rotated:
            return False
        rotated.add(log_name)
        return True
```

**When to apply**: Any shared state accessed from multiple threads, even if individual operations seem atomic.

#### Pattern 1.2: Preserve Handler State During Resource Operations

**Problem**: Calling parent class cleanup methods may invalidate more state than intended.

```python
# BEFORE: Invalidates handler state
def _rotate_file(self) -> None:
    self.close()  # Marks handler as closed!
    # ... rotate files ...
    self.stream = self._open()  # Handler still marked closed
```

```python
# AFTER: Close only the stream
def _rotate_file(self) -> None:
    if self.stream:
        self.stream.close()
        self.stream = None
    # ... rotate files ...
    self.stream = self._open()
```

**When to apply**: Resource cleanup in classes that inherit from framework base classes.

#### Pattern 1.3: Explicit Async Exception Ordering

**Problem**: `asyncio.CancelledError` is a `BaseException`, not `Exception`, but explicit handling improves clarity.

```python
# BEFORE: Implicit (works but unclear)
except Exception as e:
    logger.error(f"Task failed: {e}")
    mark_failed(task_id)
    raise
```

```python
# AFTER: Explicit cancellation handling
except asyncio.CancelledError:
    logger.info(f"Task cancelled - checkpoint preserved")
    # Don't mark as failed - allows resumption
    raise
except Exception as e:
    logger.error(f"Task failed: {e}")
    mark_failed(task_id)
    raise
```

**When to apply**: Any async code that needs different behavior for cancellation vs failure.

### Layer 2: Clarity

Document constraints and non-obvious design decisions to prevent future regressions.

**Principle**: *Intent specifications prevent maintainers from "fixing" deliberate design choices.*

#### Pattern 2.1: Document Cache Constraints

```python
# IMPORTANT: This cache assumes logger names follow the standard __name__ pattern.
# Do NOT create loggers with dynamic names (e.g., f"module.{task_id}") as the cache
# is unbounded. Module names are finite and determined by the codebase structure.
_module_log_cache: dict[str, str] = {}
```

**When to apply**: Any unbounded cache or data structure with implicit usage constraints.

#### Pattern 2.2: Document Blocking I/O in Async Contexts

```python
"""Module-based logging handlers.

Note on Async Contexts:
    These handlers perform synchronous file I/O (write, flush, unlink, rename).
    While this technically blocks the event loop, the impact is typically
    100-1000 microseconds (0.1-1ms) per operation, which is acceptable for most
    workloads where logging is not in the critical path. For high-throughput
    async applications with strict latency requirements, use QueueHandler.
"""
```

**When to apply**: Any sync I/O operations that may be called from async contexts.

#### Pattern 2.3: Document Security Properties

```python
def _compute_log_name(module_name: str) -> str:
    """Find longest matching prefix in MODULE_TO_LOG.

    SECURITY: Unmapped modules fall back to "misc" which prevents
    arbitrary file creation via malicious logger names. Never use
    module_name directly in file paths without sanitization.
    """
```

**When to apply**: Any code where security depends on current implementation details.

### Layer 3: Maintainability

Reduce cognitive load through minimal APIs, single sources of truth, and consistent organization.

**Principle**: *Less surface area = fewer bugs and easier maintenance.*

#### Pattern 3.1: Extract Shared Logic (DRY)

```python
# BEFORE: Duplicated in two handlers
class ModuleDispatchHandler:
    def _rotate_file(self, log_name: str) -> None:
        # 15 lines of rotation logic

class ThirdPartyHandler:
    def _rotate_file(self) -> None:
        # Same 15 lines duplicated
```

```python
# AFTER: Shared function
def _rotate_log_file(log_dir: Path, log_name: str, stream: TextIO | None) -> TextIO:
    """Rotate log file and return new file handle."""
    if stream:
        stream.close()
    # ... rotation logic ...
    return current.open("a", encoding="utf-8")

class ModuleDispatchHandler:
    def _rotate_file(self, log_name: str) -> None:
        self._file_cache[log_name] = _rotate_log_file(...)

class ThirdPartyHandler:
    def _rotate_file(self) -> None:
        self.stream = _rotate_log_file(...)
```

#### Pattern 3.2: Minimize Public API Surface

```python
# BEFORE: 7 exports, some internal
__all__ = [
    "start_run", "end_run", "get_current_run_id",
    "module_to_log_name", "MODULE_TO_LOG",
    "ModuleDispatchHandler", "ThirdPartyHandler",
]

# AFTER: 4 essential exports with section comments
__all__ = [
    # Run lifecycle
    "start_run",
    "end_run",
    # Handlers (for config.py)
    "ModuleDispatchHandler",
    "ThirdPartyHandler",
]
```

#### Pattern 3.3: Environment-Based Test Isolation

```python
@pytest.fixture(autouse=True)
def logging_run(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Rotate logs at test module boundaries with xdist isolation."""
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id:
        os.environ["THALA_LOG_DIR"] = f"logs/test-{worker_id}"
    # ... rest of fixture
```

**When to apply**: Any test fixture that creates files when using pytest-xdist.

## Code Review Checklist

Use this checklist when reviewing infrastructure code:

### Layer 1: Correctness
- [ ] Are there check-then-act patterns on shared state? → Add synchronization
- [ ] Do resource cleanup methods affect parent class state? → Use targeted cleanup
- [ ] Is async cancellation handled differently from exceptions? → Add explicit handler

### Layer 2: Clarity
- [ ] Are there implicit constraints on how APIs can be used? → Document them
- [ ] Are there blocking operations that could be called from async? → Document impact
- [ ] Do security properties depend on implementation details? → Document explicitly

### Layer 3: Maintainability
- [ ] Is there duplicated logic between similar classes? → Extract shared function
- [ ] Are all public exports genuinely needed externally? → Reduce `__all__`
- [ ] Are exports organized with section comments? → Add for consistency
- [ ] Do fixtures/utilities have proper type hints? → Add for IDE support

## Trade-offs

| Decision | Choice | Trade-off |
|----------|--------|-----------|
| Sync logging I/O | Kept synchronous | +Simplicity, -Potential blocking |
| Rotation lock scope | Per-operation | +Prevents races, -Slight contention |
| Cache bounds | Finite by convention | +Simple, -Must document assumption |
| API exports | Minimal (4 symbols) | +Clean API, -Tests import directly |

## Related Documentation

- [Logging Guide](/docs/guides/logging.md) - User-facing logging configuration
- [Centralized Logging Pattern](/docs/patterns/observability/centralized-logging-configuration.md) - Architecture overview
- [Task Queue Interruption Recovery](/docs/solutions/workflow-reliability/task-queue-interruption-recovery.md) - Related async patterns

## Origin

This pattern emerged from resolving 11 code review findings in commit `2cb300c` on the `feat/task-queue-interruption` branch. The findings spanned thread safety, bug fixes, async handling, test isolation, and documentation improvements.
