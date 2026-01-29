---
name: centralized-logging-configuration
title: "Centralized Logging with Two-Tier Configuration"
date: 2026-01-13
category: observability
applicability:
  - "Multi-module Python projects with multiple entry points"
  - "Applications needing production observability without excessive noise"
  - "Systems where third-party library logs must be segregated"
  - "Workflows requiring different console vs file verbosity"
components: [configuration, logging, entry_point, rotation]
complexity: simple
verified_in_production: true
related_solutions: []
tags: [logging, configuration, observability, environment-based, third-party-isolation, file-rotation, log-levels]
---

# Centralized Logging with Two-Tier Configuration

## Intent

Provide application-wide logging infrastructure with independent console and file verbosity, automatic third-party log segregation, and environment-based configuration.

## Problem

Multi-module Python applications face logging challenges:

1. **Console noise**: INFO logs overwhelm terminal during normal operation
2. **File verbosity**: DEBUG logs needed for troubleshooting but bloat main logs
3. **Third-party noise**: Libraries like httpx, elasticsearch flood logs
4. **Configuration sprawl**: Each module configures logging independently
5. **Log cleanup**: Old log files accumulate without management
6. **Async issues**: Some loggers cause race conditions during cleanup

## Solution

Centralize logging configuration with:
- Two-tier output (console + file with independent levels)
- Third-party log segregation to separate files
- Environment-based configuration (no code changes needed)
- Automatic log rotation and cleanup
- Idempotent configuration function

### Architecture

```
Application Code
    ├─ logger = logging.getLogger(__name__)
    │   └─ logs to root logger (DEBUG level)
    │
Root Logger (DEBUG)
    ├─ [Console Handler] → WARNING filter → stderr (compact format)
    │
    ├─ [Main File Handler] → INFO filter → thala.log (detailed format)
    │   └─ Receives: application loggers + langchain/langgraph
    │
    └─ Third-Party Loggers (httpx, openai, etc.)
       └─ [Third-Party Handler] → DEBUG → thala-3p.log
          └─ propagate=False (isolated from root)

Log Rotation (on startup)
    logs/
    ├── thala.log (current)
    ├── thala-3p.log (current)
    └── previous/
        ├── thala.1.log through thala.4.log
        └── thala-3p.1.log through thala-3p.4.log
```

## Implementation

### Step 1: Core Configuration Function

```python
# core/config.py

import logging
import os
from pathlib import Path

DEFAULT_CONSOLE_LEVEL = "WARNING"
DEFAULT_FILE_LEVEL = "INFO"

_logging_configured = False


def configure_logging(name: str = "thala") -> Path:
    """Configure application-wide logging with console and file handlers.

    Sets up dual logging:
    - Console: Compact format, defaults to WARNING level
    - File: Detailed format with timestamps, defaults to INFO level

    Third-party library logs are segregated to a separate file.

    Args:
        name: Base name for log files (default: "thala").
              Creates {name}.log and {name}-3p.log

    Environment variables:
        THALA_LOG_LEVEL_CONSOLE: Console log level (default: WARNING)
        THALA_LOG_LEVEL_FILE: File log level (default: INFO)
        THALA_LOG_DIR: Directory for log files (default: ./logs/)

    Returns:
        Path to the main log file

    Note:
        Safe to call multiple times (idempotent).
    """
    global _logging_configured

    console_level = os.getenv("THALA_LOG_LEVEL_CONSOLE", DEFAULT_CONSOLE_LEVEL).upper()
    file_level = os.getenv("THALA_LOG_LEVEL_FILE", DEFAULT_FILE_LEVEL).upper()
    log_dir = Path(os.getenv("THALA_LOG_DIR", _get_project_root() / "logs"))

    main_log_file = log_dir / f"{name}.log"

    if _logging_configured:
        return main_log_file

    log_dir.mkdir(parents=True, exist_ok=True)

    # Rotate existing logs
    _rotate_log(log_dir, f"{name}.log", keep=4)
    _rotate_log(log_dir, f"{name}-3p.log", keep=4)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    # Formatters
    console_formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler (compact, default WARNING)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level, logging.WARNING))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Main file handler (detailed, default INFO)
    main_file_handler = logging.FileHandler(main_log_file, mode="w", encoding="utf-8")
    main_file_handler.setLevel(getattr(logging, file_level, logging.INFO))
    main_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(main_file_handler)

    # Third-party file handler (segregated)
    third_party_log_file = log_dir / f"{name}-3p.log"
    third_party_handler = logging.FileHandler(
        third_party_log_file, mode="w", encoding="utf-8"
    )
    third_party_handler.setLevel(logging.DEBUG)
    third_party_handler.setFormatter(file_formatter)

    # Configure third-party loggers to use separate file
    for logger_name in THIRD_PARTY_LOGGERS:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.handlers.clear()
        third_party_logger.addHandler(third_party_handler)
        third_party_logger.propagate = False  # Don't send to root

    # Suppress noisy loggers that cause async race conditions
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    _logging_configured = True
    return main_log_file
```

### Step 2: Third-Party Logger Lists

```python
# Loggers that get their own file (reduce main log noise)
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
]

# Loggers that cause race conditions in async cleanup
NOISY_LOGGERS = [
    "httpcore",
    "httpcore._trace",
    "httpcore.http11",
    "httpcore.connection",
    "httpcore._async.http11",
    "httpcore._async.connection",
    "httpcore._async.connection_pool",
]
```

**Note:** `langchain` and `langgraph` are intentionally kept in main logs for workflow visibility.

### Step 3: Log Rotation

```python
def _rotate_log(log_dir: Path, base_name: str, keep: int = 4) -> None:
    """Rotate log file: current -> previous/name.1.log, shift .1->.2, etc.

    Args:
        log_dir: Directory containing the log file
        base_name: Name of the log file (e.g., "thala.log")
        keep: Number of previous versions to keep (default: 4)
    """
    previous_dir = log_dir / "previous"
    current = log_dir / base_name

    if not current.exists():
        return

    previous_dir.mkdir(exist_ok=True)

    # Parse stem and suffix
    if "." in base_name:
        stem, suffix = base_name.rsplit(".", 1)
    else:
        stem, suffix = base_name, "log"

    # Delete oldest, shift others (.4 deleted, .3->.4, .2->.3, .1->.2)
    for i in range(keep, 0, -1):
        src = previous_dir / f"{stem}.{i}.{suffix}"
        dst = previous_dir / f"{stem}.{i + 1}.{suffix}"
        if i == keep and src.exists():
            src.unlink()
        elif src.exists():
            src.rename(dst)

    # Move current to .1
    current.rename(previous_dir / f"{stem}.1.{suffix}")
```

### Step 4: Logging Level Standards

Apply these conventions consistently across the codebase:

| Level | Use For | Examples |
|-------|---------|----------|
| **DEBUG** | Internal flow, cache hits/misses, individual progress | `logger.debug(f"Cache hit for {key}")` |
| **INFO** | Workflow milestones, significant operations, summary counts | `logger.info(f"Processing {count} papers")` |
| **WARNING** | Recoverable issues, fallbacks, missing optional data | `logger.warning("Primary search failed, using fallback")` |
| **ERROR** | Critical failures with context | `logger.error(f"API call failed: {e}")` |

**Key Guidelines:**
- Move routine success paths from INFO to DEBUG
- Remove sensitive URLs from log messages
- Add context to error messages (query terms, IDs, counts)
- Standardize log message formatting

### Step 5: Usage Patterns

**Entry Points (Services, CLI, Tests):**
```python
from core.config import configure_logging

configure_logging()  # Default name "thala"
# Or with custom name for isolation:
configure_logging("academic_lit_review")

logger = logging.getLogger(__name__)
```

**Regular Modules (No configuration needed):**
```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Internal details")
logger.info("Significant operations")
logger.warning("Recoverable issues")
logger.error("Critical failures")
```

## Configuration

Environment variables for runtime control:

```bash
# Default: WARNING to console, INFO to file
python script.py

# Debug mode: See everything
THALA_LOG_LEVEL_CONSOLE=DEBUG THALA_LOG_LEVEL_FILE=DEBUG python script.py

# Verbose console for interactive debugging
THALA_LOG_LEVEL_CONSOLE=INFO python script.py

# Custom log directory
THALA_LOG_DIR=/var/log/thala python script.py
```

## Consequences

### Benefits

- **Configuration-driven**: No code changes to adjust verbosity
- **Noise reduction**: Third-party logs isolated from main application logs
- **Two-tier output**: Terminal feedback (WARNING) + file detail (INFO)
- **Automatic cleanup**: Old logs rotated automatically (keeps 5 total)
- **Idempotent**: Safe to call multiple times without reconfiguring
- **Async-safe**: Suppresses loggers that cause race conditions
- **Workflow visibility**: LangChain/LangGraph logs kept in main for tracing

### Trade-offs

- **Startup cost**: Log rotation happens at configuration time
- **File mode "w"**: Current log overwritten each run (previous versions preserved)
- **Global state**: `_logging_configured` flag manages idempotency
- **Third-party list maintenance**: New libraries may need to be added

## Related Patterns

- [Conditional Development Tracing](../llm-interaction/conditional-development-tracing.md) - Environment-based LangSmith tracing
- [Centralized Environment Configuration](../stores/centralized-env-config.md) - Environment variable patterns
- [HTTP Client Cleanup Registry](../../solutions/async-issues/http-client-cleanup-registry.md) - Resource cleanup logging

## Known Uses

- `core/config.py`: Main `configure_logging()` implementation
- `mcp_server/server.py`: MCP server entry point
- `testing/test_academic_lit_review.py`: Test script with custom log name
- `testing/test_book_finding.py`: Test script entry point
- All workflow entry points in `workflows/*/graph/api.py`

## References

- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
