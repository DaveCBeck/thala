"""Guards for LLM calls that must preserve or grow a document's length.

Two patterns are covered:

* **Continuation on max_tokens**: when a text-mode call terminates with
  ``stop_reason == "max_tokens"``, re-prompt the model to continue its own
  partial output. Concatenate parts.

* **Shrinkage floor**: after the call (or its continuations) returns, check
  that the output is at least ``shrink_ratio * len(input_content)``. Retry
  the whole call up to ``max_retries`` times. If still below floor, raise
  :class:`IntegrationShrinkageError` — callers must let this propagate so
  the task fails rather than producing a silently-truncated document.

Both patterns are idempotent: the helpers themselves do not touch workflow
state. The caller handles state updates after a successful return.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTINUATIONS = 3
_DEFAULT_MAX_RETRIES = 2  # i.e. 3 attempts total
_DEFAULT_SHRINK_RATIO = 0.8


class IntegrationShrinkageError(RuntimeError):
    """Raised when an integrator shrinks its input below the configured floor.

    Callers must let this propagate — the task should fail rather than
    continue downstream from a corrupt (summarised or truncated) document.
    """


def _extract_text(response: Any) -> str:
    """Best-effort extraction of the text payload from an AIMessage."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
            if isinstance(block, str):
                return block
    return str(content)


def _build_continuation_prompt(original_user: str, partial: str) -> str:
    tail = partial[-400:]
    return (
        original_user
        + "\n\n<Your Partial Output (truncated due to max_tokens)>\n"
        + partial
        + "\n</Your Partial Output>\n\n"
        "Your previous response was truncated. Continue writing from EXACTLY "
        "where it cut off. Do NOT repeat any content already present in your "
        "partial output. Do NOT add a preamble, header, or summary. Start "
        "your next token as a direct continuation of this tail:\n"
        f"...{tail}"
    )


async def call_with_continuation(
    *,
    tier: ModelTier,
    system: str,
    user: str,
    config: InvokeConfig | None = None,
    max_continuations: int = _DEFAULT_MAX_CONTINUATIONS,
    label: str = "integrator",
) -> tuple[str, str | None]:
    """Invoke a text-mode LLM call, stitching continuations on max_tokens.

    Returns:
        ``(collected_output, final_stop_reason)``. ``final_stop_reason``
        of ``"max_tokens"`` here means we hit the continuation budget
        without the model terminating naturally.
    """
    collected = ""
    current_user = user
    last_stop: str | None = None

    for cont in range(max_continuations + 1):
        response: AIMessage = await invoke(
            tier=tier, system=system, user=current_user, config=config
        )
        part = _extract_text(response)
        meta = getattr(response, "response_metadata", None) or {}
        last_stop = meta.get("stop_reason")

        collected = part if cont == 0 else collected + part

        if last_stop != "max_tokens":
            return collected, last_stop

        logger.warning(
            f"[{label}] hit max_tokens at {len(collected)} chars "
            f"(continuation {cont + 1}/{max_continuations})"
        )
        current_user = _build_continuation_prompt(user, collected)

    logger.error(
        f"[{label}] still truncated after {max_continuations} continuations "
        f"({len(collected)} chars collected)"
    )
    return collected, last_stop


async def with_shrinkage_guard(
    call_fn: Callable[[], Awaitable[tuple[Any, str, str | None]]],
    *,
    input_chars: int,
    shrink_ratio: float = _DEFAULT_SHRINK_RATIO,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    label: str = "integrator",
) -> Any:
    """Retry ``call_fn`` until it produces an output meeting the shrink floor.

    ``call_fn`` is a zero-arg async factory that performs one full call (with
    its own continuation handling, if relevant) and returns
    ``(raw_result, output_text, stop_reason)``. ``raw_result`` is whatever
    the caller wants back (e.g. an ``AIMessage``, a Pydantic model, or the
    text itself); ``output_text`` is used only to measure length.

    Raises:
        IntegrationShrinkageError: if the output stays below
            ``shrink_ratio * input_chars`` across all ``max_retries + 1``
            attempts.
    """
    floor = int(input_chars * shrink_ratio)
    last_output_chars = 0
    last_stop: str | None = None

    for attempt in range(max_retries + 1):
        raw, output_text, last_stop = await call_fn()
        last_output_chars = len(output_text)
        if last_output_chars >= floor:
            return raw
        logger.warning(
            f"[{label}] output below floor "
            f"({input_chars} -> {last_output_chars} chars, floor {floor}, "
            f"stop_reason={last_stop}) attempt {attempt + 1}/{max_retries + 1}"
        )

    raise IntegrationShrinkageError(
        f"[{label}] output shrank below floor on all {max_retries + 1} attempts "
        f"(input={input_chars} chars, last_output={last_output_chars} chars, "
        f"floor={floor}, last_stop={last_stop})"
    )


async def call_text_with_guards(
    *,
    input_content: str,
    tier: ModelTier,
    system: str,
    user: str,
    config: InvokeConfig | None = None,
    shrink_ratio: float = _DEFAULT_SHRINK_RATIO,
    max_continuations: int = _DEFAULT_MAX_CONTINUATIONS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    label: str = "integrator",
) -> str:
    """Convenience wrapper for text-mode integrators.

    Applies both continuation-on-max_tokens and shrinkage-floor retry, then
    returns the final text output. Raises :class:`IntegrationShrinkageError`
    on unrecoverable shrinkage.
    """

    async def _one_attempt() -> tuple[str, str, str | None]:
        text, stop = await call_with_continuation(
            tier=tier,
            system=system,
            user=user,
            config=config,
            max_continuations=max_continuations,
            label=label,
        )
        return text, text, stop

    return await with_shrinkage_guard(
        _one_attempt,
        input_chars=len(input_content),
        shrink_ratio=shrink_ratio,
        max_retries=max_retries,
        label=label,
    )


__all__ = [
    "IntegrationShrinkageError",
    "call_text_with_guards",
    "call_with_continuation",
    "with_shrinkage_guard",
    "BatchPolicy",
]
