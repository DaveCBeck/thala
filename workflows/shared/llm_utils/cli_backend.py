"""Claude Code CLI backend for invoke().

Routes LLM calls through `claude -p` subprocess, using subscription
billing instead of API billing. Enabled via THALA_LLM_BACKEND=cli.
"""

import asyncio
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Type, TypeVar

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .models import ModelTier

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# claude -p subprocess timeout — generous to allow large integrations
# (e.g. 2x12k token synthesis), but prevents indefinite hangs from
# pipe-inheritance deadlocks where child processes hold pipes open.
_CLI_TIMEOUT_SECONDS = int(os.getenv("THALA_CLI_TIMEOUT", "2700"))

# Retry count for transient failures (timeouts, non-zero exit)
_CLI_MAX_RETRIES = int(os.getenv("THALA_CLI_RETRIES", "4"))

# Cap how long we'll sleep for a subscription rate limit before giving up.
# Default 6h covers a 5-hour session window with slack; weekly limits reset
# multiple days out and should fail fast rather than sleep indefinitely.
_CLI_RATE_LIMIT_MAX_WAIT = int(os.getenv("THALA_CLI_RATE_LIMIT_MAX_WAIT", "21600"))

# Seconds past the reported reset epoch before we wake up and retry,
# to avoid racing the reset clock.
_CLI_RATE_LIMIT_BUFFER = 30

# Fallback sleep when the marker is present but no epoch can be parsed.
_CLI_RATE_LIMIT_DEFAULT_WAIT = 3600

# Matches the two forms Claude Code emits when a subscription limit trips:
#   "Claude AI usage limit reached|<unix_epoch>"
#   "Claude AI usage limit reached|<unix_epoch>|<limit_type>"   (5h / weekly / opus)
# The epoch is unix seconds. Captured so we can sleep until reset.
_USAGE_LIMIT_EPOCH_RE = re.compile(r"Claude AI usage limit reached\|(\d+)")

# Fallback for the variant with no epoch ("Your limit will reset later.").
_USAGE_LIMIT_PLAIN_RE = re.compile(r"Claude (?:AI )?usage limit reached", re.IGNORECASE)

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


class _RateLimitError(Exception):
    """Raised when claude -p surfaces a subscription usage limit.

    Carries the reset epoch (unix seconds, UTC) when parseable so the
    retry loop can sleep until the limit window reopens instead of
    burning through the short transient-error backoff.
    """

    def __init__(self, reset_epoch: float | None, raw: str):
        self.reset_epoch = reset_epoch
        self.raw = raw
        super().__init__(
            f"Claude subscription limit reached (reset_epoch={reset_epoch})"
        )


def _check_rate_limit(text: str) -> None:
    """Raise _RateLimitError if text contains a subscription-limit marker."""
    if not text:
        return
    m = _USAGE_LIMIT_EPOCH_RE.search(text)
    if m:
        raise _RateLimitError(reset_epoch=float(m.group(1)), raw=text[:500])
    if _USAGE_LIMIT_PLAIN_RE.search(text):
        raise _RateLimitError(reset_epoch=None, raw=text[:500])


async def _sleep_for_rate_limit(err: _RateLimitError, label: str) -> None:
    """Sleep until the reset epoch (clamped), logging at the boundaries."""
    if err.reset_epoch is not None:
        wait = max(0.0, err.reset_epoch - time.time()) + _CLI_RATE_LIMIT_BUFFER
        reset_str = datetime.fromtimestamp(err.reset_epoch, tz=timezone.utc).isoformat()
    else:
        wait = float(_CLI_RATE_LIMIT_DEFAULT_WAIT)
        reset_str = "unknown (no epoch in message)"

    if wait > _CLI_RATE_LIMIT_MAX_WAIT:
        logger.error(
            "CLI backend [%s]: rate-limit wait %.0fs exceeds cap %ds "
            "(reset=%s, raw=%r) — giving up",
            label, wait, _CLI_RATE_LIMIT_MAX_WAIT, reset_str, err.raw,
        )
        raise err

    logger.warning(
        "CLI backend [%s]: subscription limit reached, sleeping %.0fs until %s",
        label, wait, reset_str,
    )
    await asyncio.sleep(wait)
    logger.info("CLI backend [%s]: waking after rate-limit sleep, retrying", label)


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

    stderr_text = stderr_bytes.decode(errors="replace")
    stdout_text = stdout_bytes.decode(errors="replace")

    # Detect subscription rate limits before surfacing rc errors — the
    # marker can appear in stderr (rc != 0) OR inside the JSON result
    # field with rc == 0 and is_error=true.
    _check_rate_limit(stderr_text)
    _check_rate_limit(stdout_text)

    if rc != 0:
        raise RuntimeError(
            f"claude -p failed (rc={rc}): {stderr_text[:500] or stdout_text[:300]}"
        )

    try:
        envelope = json.loads(stdout_text)
    except json.JSONDecodeError as e:
        # Truncated/malformed stdout is a transient subprocess flake (pipe
        # buffering, SIGPIPE, partial write). Log enough context to diagnose
        # and re-raise so invoke_via_cli's retry loop handles it.
        logger.warning(
            "claude -p stdout not valid JSON (%s): len=%d, head=%r, tail=%r",
            e, len(stdout_text), stdout_text[:200], stdout_text[-200:],
        )
        raise
    # Defensive: also inspect the parsed result field explicitly in case
    # future Claude Code versions escape the pipe character in raw stdout.
    if isinstance(envelope, dict):
        result_field = envelope.get("result")
        if isinstance(result_field, str):
            _check_rate_limit(result_field)
    return envelope


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


def _envelope_to_message(envelope: dict, text: str, model: str) -> AIMessage:
    """Build an AIMessage from a Claude Code JSON envelope.

    Propagates stop_reason, usage, and model into response_metadata and
    usage_metadata so callers can detect truncation (max_tokens) and
    measure cost, matching the broker path's metadata shape.
    """
    stop_reason = envelope.get("stop_reason")
    usage_raw = envelope.get("usage") or {}
    input_tokens = usage_raw.get("input_tokens", 0) or 0
    output_tokens = usage_raw.get("output_tokens", 0) or 0

    usage: dict = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if "cache_creation_input_tokens" in usage_raw:
        usage["cache_creation_input_tokens"] = usage_raw["cache_creation_input_tokens"]
    if "cache_read_input_tokens" in usage_raw:
        usage["cache_read_input_tokens"] = usage_raw["cache_read_input_tokens"]

    if stop_reason == "max_tokens":
        logger.warning(
            f"CLI response truncated (max_tokens): model={model} output_tokens={output_tokens}"
        )
    else:
        logger.debug(
            f"CLI response: model={model} stop_reason={stop_reason} output_tokens={output_tokens}"
        )

    return AIMessage(
        content=text,
        response_metadata={
            "stop_reason": stop_reason,
            "usage": usage,
            "model": envelope.get("model") or model,
        },
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


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
    attempt = 0
    while attempt < _CLI_MAX_RETRIES:
        try:
            logger.debug("CLI backend: invoking %s (text, effort=%s, attempt=%d)", model, effort, attempt + 1)
            envelope = await _run_claude_cli(cmd, user_prompt)
            text = envelope["result"]
            return _envelope_to_message(envelope, text, model)
        except _RateLimitError as e:
            # Sleep until reset; do NOT count against retry budget.
            await _sleep_for_rate_limit(e, f"invoke_via_cli[{model}]")
            continue
        except (TimeoutError, RuntimeError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            attempt += 1
            logger.warning("CLI backend: attempt %d/%d failed: %s", attempt, _CLI_MAX_RETRIES, e)
            if attempt < _CLI_MAX_RETRIES:
                await asyncio.sleep(2 ** (attempt - 1))
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
    attempt = 0
    while attempt < _CLI_MAX_RETRIES:
        try:
            logger.debug(
                "CLI backend: invoking %s (structured=%s, effort=%s, attempt=%d)",
                model, schema.__name__, effort, attempt + 1,
            )
            envelope = await _run_claude_cli(cmd, user_prompt)
            raw = envelope["structured_output"]
            return schema.model_validate(raw)
        except _RateLimitError as e:
            await _sleep_for_rate_limit(e, f"invoke_structured_via_cli[{model}]")
            continue
        except (TimeoutError, RuntimeError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            attempt += 1
            logger.warning("CLI backend: attempt %d/%d failed: %s", attempt, _CLI_MAX_RETRIES, e)
            if attempt < _CLI_MAX_RETRIES:
                await asyncio.sleep(2 ** (attempt - 1))
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MCP-based tool agent support
# ---------------------------------------------------------------------------

# Known tool names that the paper_tools MCP server can handle
_MCP_SUPPORTED_TOOLS = {"search_papers", "get_paper_content"}


def _build_mcp_config(tools: list[BaseTool]) -> str:
    """Build inline MCP config JSON for the paper tools server.

    Args:
        tools: LangChain tools requested by the caller.

    Returns:
        JSON string suitable for --mcp-config flag.

    Raises:
        ValueError: If any tool is not supported by the MCP server.
    """
    tool_names = {t.name for t in tools}
    unsupported = tool_names - _MCP_SUPPORTED_TOOLS
    if unsupported:
        raise ValueError(
            f"Tools not supported by MCP server: {unsupported}. "
            f"Supported: {_MCP_SUPPORTED_TOOLS}"
        )

    # Pass through THALA_* env vars so the MCP server can reach stores/services.
    # The server also runs load_dotenv(), but explicit env is more robust when
    # spawned as a grandchild process via Claude CLI.
    mcp_env = {
        k: v for k, v in os.environ.items()
        if k.startswith("THALA_") or k.startswith("EMBEDDING_") or k == "LANGSMITH_API_KEY"
    }
    mcp_env["THALA_MCP_TOOLS"] = ",".join(sorted(tool_names))

    config = {
        "mcpServers": {
            "paper_tools": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.paper_tools"],
                "env": mcp_env,
            }
        }
    }
    return json.dumps(config)


async def invoke_tool_agent_via_cli(
    tier: ModelTier,
    system: str,
    user_prompt: str,
    schema: Type[T],
    tools: list[BaseTool],
    *,
    effort: str | None = None,
    max_tool_calls: int = 12,
) -> T:
    """Invoke Claude with MCP tools via `claude -p` for agentic tool-calling.

    Spawns a `claude -p` subprocess with --mcp-config pointing at the
    paper_tools MCP server, which wraps the LangChain tools. Claude
    handles the multi-turn tool loop internally and returns structured
    output via --json-schema.

    Args:
        tier: Model tier (must be a Claude tier, not DeepSeek)
        system: System prompt
        user_prompt: User prompt text
        schema: Pydantic model class for structured output
        tools: LangChain tools to expose via MCP
        effort: Optional effort level (low/medium/high/max)
        max_tool_calls: Max tool calls to allow (maps to --max-turns + 8)

    Returns:
        Validated Pydantic model instance
    """
    model = _TIER_TO_CLI_MODEL[tier]
    mcp_config = _build_mcp_config(tools)
    json_schema = json.dumps(schema.model_json_schema())

    # max_turns = tool calls + 8 for structured output turn + margin
    max_turns = max_tool_calls + 8

    # Build command manually instead of _build_base_cmd because the tool
    # agent needs MCP tools enabled (no --tools "") and --allowedTools
    # to restrict to only MCP server tools.
    cmd = [
        "claude",
        "-p",
        "--output-format", "json",
        "--system-prompt", system,
        "--model", model,
        "--max-turns", str(max_turns),
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        "--mcp-config", mcp_config,
        "--strict-mcp-config",
        "--json-schema", json_schema,
    ]
    if effort:
        cmd.extend(["--effort", effort])

    last_err: Exception | None = None
    attempt = 0
    while attempt < _CLI_MAX_RETRIES:
        try:
            logger.debug(
                "CLI backend: invoking %s (tool_agent, tools=%s, schema=%s, "
                "max_turns=%d, effort=%s, attempt=%d)",
                model, [t.name for t in tools], schema.__name__,
                max_turns, effort, attempt + 1,
            )
            envelope = await _run_claude_cli(cmd, user_prompt)
            raw = envelope["structured_output"]
            return schema.model_validate(raw)
        except _RateLimitError as e:
            await _sleep_for_rate_limit(e, f"invoke_tool_agent_via_cli[{model}]")
            continue
        except (TimeoutError, RuntimeError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            attempt += 1
            logger.warning(
                "CLI backend: tool agent attempt %d/%d failed: %s",
                attempt, _CLI_MAX_RETRIES, e,
            )
            if attempt < _CLI_MAX_RETRIES:
                await asyncio.sleep(2 ** (attempt - 1))
    raise last_err  # type: ignore[misc]
