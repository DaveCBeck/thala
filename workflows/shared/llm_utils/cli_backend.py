"""Claude Code CLI backend for invoke().

Routes LLM calls through `claude -p` subprocess, using subscription
billing instead of API billing. Enabled via THALA_LLM_BACKEND=cli.
"""

import asyncio
import json
import logging
import os
from typing import Type, TypeVar

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from .models import ModelTier

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

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


async def _run_claude_cli(
    cmd: list[str],
    user_prompt: str,
) -> dict:
    """Run claude -p subprocess and return parsed JSON envelope.

    Args:
        cmd: Full command list for claude -p
        user_prompt: User prompt to pipe via stdin

    Returns:
        Parsed JSON envelope from stdout

    Raises:
        RuntimeError: If subprocess exits non-zero or output is not valid JSON
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=user_prompt.encode())

    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed (rc={proc.returncode}): {stderr.decode()[:500]}")

    return json.loads(stdout.decode())


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

    logger.debug(f"CLI backend: invoking {model} (text, effort={effort})")
    envelope = await _run_claude_cli(cmd, user_prompt)

    # claude -p --output-format json returns {"result": "text string", ...}
    text = envelope["result"]
    return AIMessage(content=text)


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

    logger.debug(f"CLI backend: invoking {model} (structured={schema.__name__}, effort={effort})")
    envelope = await _run_claude_cli(cmd, user_prompt)

    # claude -p --json-schema returns {"structured_output": {...}, ...} at top level
    raw = envelope["structured_output"]
    return schema.model_validate(raw)
