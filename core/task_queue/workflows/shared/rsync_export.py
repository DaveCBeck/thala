"""Rsync batch directories to VPS for publishing.

Reads SSH credentials from environment variables:
- VPS_HOST: hostname or IP
- VPS_USER: SSH username
- VPS_SSH_KEY_PATH: path to private key (should be in gitignored keys/ dir)
- VPS_EXPORT_PATH: remote base directory for batch folders
"""

import asyncio
import logging
import os

from pathlib import Path

logger = logging.getLogger(__name__)


async def rsync_batch(batch_dir: Path) -> bool:
    """Rsync a batch directory to the VPS.

    The batch_dir's relative path under the export root is preserved on the
    remote side (e.g. export/arrivingfuture/batch_0003/ → remote/arrivingfuture/batch_0003/).

    Writes a .complete marker file in batch_dir after successful transfer.

    Returns True on success, False on failure.
    """
    host = os.environ.get("VPS_HOST")
    user = os.environ.get("VPS_USER")
    key_path = os.environ.get("VPS_SSH_KEY_PATH")
    remote_base = os.environ.get("VPS_EXPORT_PATH")

    if not all([host, user, key_path, remote_base]):
        missing = [k for k in ("VPS_HOST", "VPS_USER", "VPS_SSH_KEY_PATH", "VPS_EXPORT_PATH") if not os.environ.get(k)]
        logger.error("Missing VPS env vars: %s", ", ".join(missing))
        return False

    # Derive remote path: {remote_base}/{pub_slug}/{batch_name}/
    # batch_dir is like .thala/export/arrivingfuture/batch_0003
    pub_slug = batch_dir.parent.name
    batch_name = batch_dir.name
    remote_path = f"{remote_base.rstrip('/')}/{pub_slug}/{batch_name}/"

    ssh_cmd = f"ssh -i {key_path} -o StrictHostKeyChecking=accept-new"

    # Ensure remote directory exists
    mkdir_proc = await asyncio.create_subprocess_exec(
        "ssh", "-i", key_path, "-o", "StrictHostKeyChecking=accept-new",
        f"{user}@{host}", f"mkdir -p {remote_path}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await mkdir_proc.communicate()

    cmd = [
        "rsync", "-avz",
        "-e", ssh_cmd,
        f"{batch_dir}/",
        f"{user}@{host}:{remote_path}",
    ]

    logger.info("Rsyncing %s → %s@%s:%s", batch_dir, user, host, remote_path)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error("rsync failed (exit %d): %s", proc.returncode, stderr.decode())
        return False

    # Write completion markers (local + remote)
    (batch_dir / ".complete").write_text("")
    touch_proc = await asyncio.create_subprocess_exec(
        "ssh", "-i", key_path, "-o", "StrictHostKeyChecking=accept-new",
        f"{user}@{host}", f"touch {remote_path}.complete",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await touch_proc.communicate()
    if touch_proc.returncode != 0:
        logger.warning("Failed to write remote .complete marker (rsync itself succeeded)")

    logger.info("rsync complete: %s", batch_dir.name)
    return True
