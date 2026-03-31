"""Tests for rsync export module."""

from unittest.mock import AsyncMock, patch

import pytest

from core.task_queue.workflows.shared.rsync_export import rsync_batch


@pytest.mark.asyncio
async def test_returns_false_when_env_vars_missing(tmp_path):
    """rsync_batch returns False when VPS env vars are not set."""
    with patch.dict("os.environ", {}, clear=True):
        result = await rsync_batch(tmp_path / "batch_0001")
    assert result is False


@pytest.mark.asyncio
async def test_constructs_correct_rsync_command(tmp_path):
    """rsync_batch passes the correct arguments to subprocess."""
    batch_dir = tmp_path / "arrivingfuture" / "batch_0003"
    batch_dir.mkdir(parents=True)

    env = {
        "VPS_HOST": "10.0.0.1",
        "VPS_USER": "deploy",
        "VPS_SSH_KEY_PATH": "/keys/id_ed25519",
        "VPS_EXPORT_PATH": "/srv/articles",
    }

    # First call (test -d) returns 1 (dir doesn't exist), rest return 0
    check_proc = AsyncMock()
    check_proc.communicate = AsyncMock(return_value=(b"", b""))
    check_proc.returncode = 1

    ok_proc = AsyncMock()
    ok_proc.communicate = AsyncMock(return_value=(b"", b""))
    ok_proc.returncode = 0

    with (
        patch.dict("os.environ", env, clear=True),
        patch(
            "core.task_queue.workflows.shared.rsync_export.asyncio.create_subprocess_exec",
            side_effect=[check_proc, ok_proc, ok_proc, ok_proc],
        ) as mock_exec,
    ):
        result = await rsync_batch(batch_dir)

    assert result is True
    assert (batch_dir / ".complete").exists()

    # Verify rsync was called with correct args (3rd call: check, mkdir, rsync, touch)
    rsync_call = mock_exec.call_args_list[2][0]
    assert rsync_call[0] == "rsync"
    assert f"{batch_dir}/" in rsync_call
    assert "deploy@10.0.0.1:/srv/articles/arrivingfuture/batch_0003/" in rsync_call


@pytest.mark.asyncio
async def test_refuses_to_overwrite_existing_remote_batch(tmp_path):
    """rsync_batch returns False when remote batch dir already exists."""
    batch_dir = tmp_path / "arrivingfuture" / "batch_0001"
    batch_dir.mkdir(parents=True)

    env = {
        "VPS_HOST": "10.0.0.1",
        "VPS_USER": "deploy",
        "VPS_SSH_KEY_PATH": "/keys/id_ed25519",
        "VPS_EXPORT_PATH": "/srv/articles",
    }

    # test -d returns 0 (dir exists)
    check_proc = AsyncMock()
    check_proc.communicate = AsyncMock(return_value=(b"", b""))
    check_proc.returncode = 0

    with (
        patch.dict("os.environ", env, clear=True),
        patch(
            "core.task_queue.workflows.shared.rsync_export.asyncio.create_subprocess_exec",
            return_value=check_proc,
        ),
    ):
        result = await rsync_batch(batch_dir)

    assert result is False
    assert not (batch_dir / ".complete").exists()


@pytest.mark.asyncio
async def test_returns_false_on_rsync_failure(tmp_path):
    """rsync_batch returns False when rsync exits non-zero."""
    batch_dir = tmp_path / "pub" / "batch_0001"
    batch_dir.mkdir(parents=True)

    env = {
        "VPS_HOST": "10.0.0.1",
        "VPS_USER": "deploy",
        "VPS_SSH_KEY_PATH": "/keys/id_ed25519",
        "VPS_EXPORT_PATH": "/srv/articles",
    }

    # Pre-flight check passes (dir doesn't exist), mkdir ok, rsync fails
    check_proc = AsyncMock()
    check_proc.communicate = AsyncMock(return_value=(b"", b""))
    check_proc.returncode = 1

    mkdir_proc = AsyncMock()
    mkdir_proc.communicate = AsyncMock(return_value=(b"", b""))
    mkdir_proc.returncode = 0

    fail_proc = AsyncMock()
    fail_proc.communicate = AsyncMock(return_value=(b"", b"connection refused"))
    fail_proc.returncode = 1

    with (
        patch.dict("os.environ", env, clear=True),
        patch(
            "core.task_queue.workflows.shared.rsync_export.asyncio.create_subprocess_exec",
            side_effect=[check_proc, mkdir_proc, fail_proc],
        ),
    ):
        result = await rsync_batch(batch_dir)

    assert result is False
    assert not (batch_dir / ".complete").exists()
