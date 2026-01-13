"""Unified token utilities for accurate budget management.

Provides both quick character-based estimates (with safety margins) and
accurate tiktoken-based counting for critical path decisions.

This module consolidates token management that was previously duplicated
across loop5_factcheck.py and core/embedding.py.
"""

import logging
from functools import lru_cache
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)

# =============================================================================
# Model Context Limits
# =============================================================================
# Maximum tokens per model (theoretical limits)
HAIKU_MAX_TOKENS = 200_000
SONNET_MAX_TOKENS = 200_000
SONNET_1M_MAX_TOKENS = 1_000_000
OPUS_MAX_TOKENS = 200_000

# Safe operational limits (leave headroom for response + overhead)
# These are conservative limits to prevent hitting hard caps
HAIKU_SAFE_LIMIT = 150_000  # 75% of max
SONNET_SAFE_LIMIT = 150_000  # 75% of max
SONNET_1M_SAFE_LIMIT = 800_000  # 80% of max
OPUS_SAFE_LIMIT = 150_000  # 75% of max

# =============================================================================
# Estimation Constants
# =============================================================================
CHARS_PER_TOKEN = 4  # Conservative estimate for Claude tokenizer
SAFETY_MARGIN = 0.30  # 30% buffer for JSON overhead, tool definitions, etc.
JSON_OVERHEAD_FACTOR = 1.10  # ~10% overhead for JSON formatting
TOOL_DEFINITION_BUFFER = 2000  # Approximate tokens for tool schemas
MESSAGE_OVERHEAD_PER_TURN = 50  # Tokens per message wrapper (role, metadata)

# Response buffers
DEFAULT_RESPONSE_BUFFER = 4096
EXTENDED_RESPONSE_BUFFER = 16384


# =============================================================================
# Token Counting Functions
# =============================================================================


@lru_cache(maxsize=1)
def _get_encoding():
    """Get tiktoken encoding (cached for performance).

    Uses cl100k_base which is compatible with Claude's tokenizer.
    """
    return tiktoken.get_encoding("cl100k_base")


def count_tokens_accurate(text: str) -> int:
    """Count tokens accurately using tiktoken.

    Use for critical path decisions (pre-flight checks, model selection).
    ~10x slower than estimate_tokens_fast but precise.

    Args:
        text: Text to count tokens for

    Returns:
        Exact token count using cl100k_base encoding
    """
    if not text:
        return 0
    encoding = _get_encoding()
    return len(encoding.encode(text))


def estimate_tokens_fast(text: str, with_safety_margin: bool = True) -> int:
    """Quick token estimate using character count.

    Use for budget tracking during execution where speed matters.
    Includes safety margin by default for conservative estimates.

    Args:
        text: Text to estimate tokens for
        with_safety_margin: Apply 30% safety margin (default True)

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    base_estimate = len(text) // CHARS_PER_TOKEN
    if with_safety_margin:
        return int(base_estimate * (1 + SAFETY_MARGIN))
    return base_estimate


def estimate_request_tokens(
    user_prompt: str,
    system_prompt: str = "",
    tool_results: str = "",
    message_count: int = 0,
    include_tool_definitions: bool = False,
    response_buffer: int = DEFAULT_RESPONSE_BUFFER,
) -> int:
    """Estimate total tokens for an LLM request.

    Accounts for:
    - User and system prompt content
    - Tool result content from previous turns
    - Per-message overhead (JSON wrappers)
    - Tool definition schemas (if tools enabled)
    - Response buffer for model output
    - Safety margin for estimation error

    Args:
        user_prompt: User message content
        system_prompt: System message content
        tool_results: Accumulated tool result content
        message_count: Number of messages in history
        include_tool_definitions: Whether tools are bound to LLM
        response_buffer: Reserved tokens for response

    Returns:
        Conservative token estimate
    """
    # Base content tokens (without safety margin initially)
    content_tokens = estimate_tokens_fast(
        user_prompt + system_prompt + tool_results,
        with_safety_margin=False,
    )

    # Add JSON formatting overhead
    content_tokens = int(content_tokens * JSON_OVERHEAD_FACTOR)

    # Add per-message overhead
    message_overhead = message_count * MESSAGE_OVERHEAD_PER_TURN

    # Add tool definitions if applicable
    tool_overhead = TOOL_DEFINITION_BUFFER if include_tool_definitions else 0

    # Total with safety margin
    subtotal = content_tokens + message_overhead + tool_overhead + response_buffer
    return int(subtotal * (1 + SAFETY_MARGIN))


# =============================================================================
# Budget Checking
# =============================================================================


class TokenBudgetExceeded(Exception):
    """Raised when a request would exceed token budget."""

    def __init__(
        self,
        estimated_tokens: int,
        limit: int,
        message: Optional[str] = None,
    ):
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        msg = message or f"Estimated {estimated_tokens:,} tokens exceeds safe limit {limit:,}"
        super().__init__(msg)


def check_token_budget(
    estimated_tokens: int,
    limit: int,
    threshold: float = 0.95,
    raise_on_exceed: bool = True,
) -> bool:
    """Check if estimated tokens are within budget.

    Args:
        estimated_tokens: Estimated token count
        limit: Token limit to check against
        threshold: Fraction of limit that triggers warning/error (default 95%)
        raise_on_exceed: If True, raise TokenBudgetExceeded; else return False

    Returns:
        True if within budget, False if exceeded (when raise_on_exceed=False)

    Raises:
        TokenBudgetExceeded: If over budget and raise_on_exceed=True
    """
    threshold_limit = int(limit * threshold)

    if estimated_tokens > threshold_limit:
        if raise_on_exceed:
            raise TokenBudgetExceeded(
                estimated_tokens=estimated_tokens,
                limit=limit,
                message=(
                    f"Estimated {estimated_tokens:,} tokens exceeds "
                    f"{threshold:.0%} threshold ({threshold_limit:,}) of limit {limit:,}"
                ),
            )
        return False
    return True


def get_safe_limit_for_model(model_id: str) -> int:
    """Get safe token limit for a model.

    Returns conservative limit leaving headroom for response and overhead.

    Args:
        model_id: Model identifier (e.g., "claude-haiku-4-5-20251001")

    Returns:
        Safe operational token limit
    """
    model_lower = model_id.lower()
    if "sonnet" in model_lower:
        # Check for 1M context variant
        return SONNET_1M_SAFE_LIMIT if "1m" in model_lower else SONNET_SAFE_LIMIT
    elif "opus" in model_lower:
        return OPUS_SAFE_LIMIT
    else:  # haiku or default
        return HAIKU_SAFE_LIMIT


# =============================================================================
# Model Selection
# =============================================================================


def select_model_for_context(
    estimated_tokens: int,
    prefer_haiku: bool = True,
) -> str:
    """Select appropriate model tier based on context size.

    Args:
        estimated_tokens: Estimated input tokens
        prefer_haiku: Prefer Haiku for smaller contexts (default True)

    Returns:
        Model tier string: "haiku", "sonnet", or "sonnet_1m"
    """
    if estimated_tokens <= HAIKU_SAFE_LIMIT and prefer_haiku:
        return "haiku"
    elif estimated_tokens <= SONNET_SAFE_LIMIT:
        return "sonnet"
    elif estimated_tokens <= SONNET_1M_SAFE_LIMIT:
        logger.debug(
            f"Context size {estimated_tokens:,} tokens requires SONNET_1M "
            f"(exceeds {SONNET_SAFE_LIMIT:,} safe limit)"
        )
        return "sonnet_1m"
    else:
        logger.warning(
            f"Context size {estimated_tokens:,} exceeds even 1M safe limit "
            f"({SONNET_1M_SAFE_LIMIT:,}), using sonnet_1m anyway"
        )
        return "sonnet_1m"


