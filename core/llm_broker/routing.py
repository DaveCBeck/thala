"""Request routing logic for the LLM Broker.

Determines whether a request should be routed through the batch API
or synchronous API based on user mode, call-site policy, and model tier.
"""

from core.types import ModelTier, is_deepseek_tier

from .schemas import BatchPolicy, UserMode


def should_batch(
    mode: UserMode,
    policy: BatchPolicy,
    model: ModelTier,
    thinking_budget: int | None,
) -> bool:
    """Determine if a request should be batched.

    Args:
        mode: Current user mode
        policy: Call-site batch policy
        model: Model tier
        thinking_budget: Extended thinking budget

    Returns:
        True if request should be queued for batching
    """
    # DeepSeek doesn't support batch API
    if is_deepseek_tier(model):
        return False

    # Extended thinking incompatible with batch
    if thinking_budget:
        return False

    # Policy + Mode matrix
    if policy == BatchPolicy.REQUIRE_SYNC:
        return False
    if policy == BatchPolicy.FORCE_BATCH:
        return True
    if mode == UserMode.FAST:
        return False
    if policy == BatchPolicy.PREFER_BALANCE and mode in (
        UserMode.BALANCED,
        UserMode.ECONOMICAL,
    ):
        return True
    if policy == BatchPolicy.PREFER_SPEED and mode == UserMode.ECONOMICAL:
        return True

    return False
