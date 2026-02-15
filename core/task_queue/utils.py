"""Shared utilities for core/task_queue."""

import json
import os
import tempfile
from pathlib import Path


def write_json_atomic(path: Path, data: dict, *, indent: int | None = None) -> None:
    """Write *data* as JSON to *path* atomically (temp file + rename).

    The temp file is created in the same directory as *path* so that the
    final ``Path.rename()`` is a same-filesystem atomic operation.

    Args:
        path: Destination file path.
        data: JSON-serialisable dict to write.
        indent: Optional indentation level passed to ``json.dump``.
    """
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
        Path(tmp_path).rename(path)
    except Exception:
        # Clean up temp file on error
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise
