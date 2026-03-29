"""Claude Code CLI backend for invoke().

Routes LLM calls through `claude -p` subprocess, using subscription
billing instead of API billing. Enabled via THALA_LLM_BACKEND=cli.
"""

import asyncio
import json
import logging
import os
import signal
from typing import Type, TypeVar

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from .models import ModelTier

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# claude -p subprocess timeout — generous to allow large integrations
# (e.g. 2x12k token synthesis), but prevents indefinite hangs from
# pipe-inheritance deadlocks where child processes hold pipes open.
_CLI_TIMEOUT_SECONDS = int(os.getenv("THALA_CLI_TIMEOUT", "1200"))

# Retry count for transient failures (timeouts, non-zero exit)
_CLI_MAX_RETRIES = int(os.getenv("THALA_CLI_RETRIES", "4"))

# Model tier → claude -p --model value
_TIER_TO_CLI_MODEL: dict[ModelTier, str] = {
    ModelTier.HAIKU: "haiku",
    ModelTier.SONNET: "sonnet",
    ModelTier.SONNET_1M: "sonnet",  # Claude Code manages context internally
    ModelTier.OPUS: "opus",
}


def is_cli_backend_enabled() -> bool:
    """Check if the CLI backend is enabled via environment variable."""
    return os.getenv("THALA_LLM_BACKEND", "api").lower() == "cli"


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill subprocess and all its children, then reap synchronously."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass
    # Reap via synchronous waitpid — asyncio's proc.wait() can hang
    # when the pidfd child watcher loses track of a process.
    try:
        os.waitpid(proc.pid, 0)
    except ChildProcessError:
        pass  # already reaped


async def _run_claude_cli(
    cmd: list[str],
    user_prompt: str,
) -> dict:
    """Run claude -p subprocess and return parsed JSON envelope.

    Avoids asyncio.Process.communicate() because its internal proc.wait()
    can hang when the pidfd child watcher fails to register the process
    exit (observed with start_new_session=True on Linux/WSL2). Instead,
    we read pipes directly and use synchronous waitpid for reaping.

    Args:
        cmd: Full command list for claude -p
        user_prompt: User prompt to pipe via stdin

    Returns:
        Parsed JSON envelope from stdout

    Raises:
        RuntimeError: If subprocess exits non-zero or output is not valid JSON
        TimeoutError: If subprocess does not complete within _CLI_TIMEOUT_SECONDS
    """
    # Strip ANTHROPIC_API_KEY so claude -p uses Max subscription billing
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    # start_new_session=True puts the subprocess in its own process group,
    # so we can kill it AND any children (preventing pipe-inheritance zombies).
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        start_new_session=True,
    )

    try:
        # Feed stdin and close it
        proc.stdin.write(user_prompt.encode())
        await proc.stdin.drain()
        proc.stdin.close()

        # Read stdout/stderr with a hard timeout.  We avoid communicate()
        # because it ends with `await proc.wait()` which can hang when
        # the asyncio pidfd child watcher misses the process exit.
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            asyncio.gather(proc.stdout.read(), proc.stderr.read()),
            timeout=_CLI_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        await asyncio.to_thread(_kill_process_tree, proc)
        raise TimeoutError(
            f"claude -p timed out after {_CLI_TIMEOUT_SECONDS}s "
            f"(cmd: {' '.join(cmd[:6])}...)"
        )

    # Reap synchronously — don't rely on asyncio's child watcher.
    # ChildProcessError means asyncio's watcher already reaped it;
    # fall back to proc.returncode in that case.
    try:
        _, status = await asyncio.to_thread(os.waitpid, proc.pid, 0)
        rc = os.waitstatus_to_exitcode(status)
    except ChildProcessError:
        rc = proc.returncode if proc.returncode is not None else 0

    if rc != 0:
        raise RuntimeError(
            f"claude -p failed (rc={rc}): {stderr_bytes.decode()[:500]}"
        )

    return json.loads(stdout_bytes.decode())


def _build_base_cmd(
    model: str,
    system: str,
    effort: str | None = None,
    max_turns: int = 1,
) -> list[str]:
    """Build the base claude -p command with common flags.

    Args:
        model: Claude model alias (haiku/sonnet/opus)
        system: System prompt
        effort: Optional effort level
        max_turns: Max agentic turns (1 for text, 2 for structured output
                   which needs a tool_use round-trip)
    """
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--system-prompt",
        system,
        "--model",
        model,
        "--tools",
        "",
        "--max-turns",
        str(max_turns),
        "--no-session-persistence",
        "--dangerously-skip-permissions",
    ]
    if effort:
        cmd.extend(["--effort", effort])
    return cmd


async def invoke_via_cli(
    tier: ModelTier,
    system: str,
    user_prompt: str,
    *,
    effort: str | None = None,
    max_tokens: int = 4096,
) -> AIMessage:
    """Invoke Claude via `claude -p` subprocess.

    Args:
        tier: Model tier (must be a Claude tier, not DeepSeek)
        system: System prompt
        user_prompt: User prompt text
        effort: Optional effort level (low/medium/high/max)
        max_tokens: Unused (Claude Code manages internally), kept for interface parity

    Returns:
        AIMessage with the response text
    """
    model = _TIER_TO_CLI_MODEL[tier]
    cmd = _build_base_cmd(model, system, effort)

    last_err: Exception | None = None
    for attempt in range(_CLI_MAX_RETRIES):
        try:
            logger.debug("CLI backend: invoking %s (text, effort=%s, attempt=%d)", model, effort, attempt + 1)
            envelope = await _run_claude_cli(cmd, user_prompt)
            text = envelope["result"]
            return AIMessage(content=text)
        except (TimeoutError, RuntimeError) as e:
            last_err = e
            logger.warning("CLI backend: attempt %d/%d failed: %s", attempt + 1, _CLI_MAX_RETRIES, e)
            if attempt < _CLI_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
    raise last_err  # type: ignore[misc]


async def invoke_structured_via_cli(
    tier: ModelTier,
    system: str,
    user_prompt: str,
    schema: Type[T],
    *,
    effort: str | None = None,
    max_tokens: int = 4096,
) -> T:
    """Invoke Claude via `claude -p --json-schema` for structured output.

    Args:
        tier: Model tier (must be a Claude tier, not DeepSeek)
        system: System prompt
        user_prompt: User prompt text
        schema: Pydantic model class for validated output
        effort: Optional effort level (low/medium/high/max)
        max_tokens: Unused (Claude Code manages internally), kept for interface parity

    Returns:
        Validated Pydantic model instance
    """
    model = _TIER_TO_CLI_MODEL[tier]
    json_schema = json.dumps(schema.model_json_schema())

    # --json-schema uses a tool_use round-trip, needs 2 turns
    cmd = _build_base_cmd(model, system, effort, max_turns=2)
    cmd.extend(["--json-schema", json_schema])

    last_err: Exception | None = None
    for attempt in range(_CLI_MAX_RETRIES):
        try:
            logger.debug(
                "CLI backend: invoking %s (structured=%s, effort=%s, attempt=%d)",
                model, schema.__name__, effort, attempt + 1,
            )
            envelope = await _run_claude_cli(cmd, user_prompt)
            raw = envelope["structured_output"]
            return schema.model_validate(raw)
        except (TimeoutError, RuntimeError) as e:
            last_err = e
            logger.warning("CLI backend: attempt %d/%d failed: %s", attempt + 1, _CLI_MAX_RETRIES, e)
            if attempt < _CLI_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
    raise last_err  # type: ignore[misc]
