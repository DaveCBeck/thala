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

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0

    with (
        patch.dict("os.environ", env, clear=True),
        patch("core.task_queue.workflows.shared.rsync_export.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
    ):
        result = await rsync_batch(batch_dir)

    assert result is True
    assert (batch_dir / ".complete").exists()

    # Verify rsync was called with correct args
    call_args = mock_exec.call_args[0]
    assert call_args[0] == "rsync"
    assert f"{batch_dir}/" in call_args
    assert "deploy@10.0.0.1:/srv/articles/arrivingfuture/batch_0003/" in call_args


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

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"connection refused"))
    mock_proc.returncode = 1

    with (
        patch.dict("os.environ", env, clear=True),
        patch("core.task_queue.workflows.shared.rsync_export.asyncio.create_subprocess_exec", return_value=mock_proc),
    ):
        result = await rsync_batch(batch_dir)

    assert result is False
    assert not (batch_dir / ".complete").exists()
