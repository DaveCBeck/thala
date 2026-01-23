"""
Model pricing map for cost calculation.

Pricing per million tokens (USD) from Anthropic and DeepSeek docs.
Used to calculate costs from LangSmith token counts.
"""

# Pricing per million tokens (USD)
# Source: Anthropic and DeepSeek pricing pages
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Claude models (Anthropic)
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00},
    # DeepSeek models (much cheaper)
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # OpenAI embeddings (for reference, usually negligible)
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}

# Aliases for common model name variations
MODEL_ALIASES: dict[str, str] = {
    "claude-3-5-haiku-20251001": "claude-haiku-4-5-20251001",
    "claude-3-5-sonnet-20250929": "claude-sonnet-4-5-20250929",
    "claude-3-5-opus-20251101": "claude-opus-4-5-20251101",
}


def get_model_pricing(model_name: str) -> dict[str, float]:
    """Get pricing for a model, handling aliases.

    Args:
        model_name: Model identifier from LangSmith metadata

    Returns:
        {"input": price_per_mtok, "output": price_per_mtok}
        Returns {"input": 0, "output": 0} for unknown models
    """
    # Check direct match
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # Check aliases
    if model_name in MODEL_ALIASES:
        return MODEL_PRICING[MODEL_ALIASES[model_name]]

    # Try partial matching for model families
    model_lower = model_name.lower()
    if "haiku" in model_lower:
        return MODEL_PRICING["claude-haiku-4-5-20251001"]
    if "sonnet" in model_lower:
        return MODEL_PRICING["claude-sonnet-4-5-20250929"]
    if "opus" in model_lower:
        return MODEL_PRICING["claude-opus-4-5-20251101"]
    if "deepseek" in model_lower and "reason" in model_lower:
        return MODEL_PRICING["deepseek-reasoner"]
    if "deepseek" in model_lower:
        return MODEL_PRICING["deepseek-chat"]

    # Unknown model - return zero cost
    return {"input": 0.0, "output": 0.0}


def calculate_run_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate cost for a single LLM run.

    Args:
        model_name: Model identifier
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    pricing = get_model_pricing(model_name)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def format_cost(cost_usd: float) -> str:
    """Format cost for display.

    Args:
        cost_usd: Cost in USD

    Returns:
        Formatted string like "$1.23" or "$0.0012"
    """
    if cost_usd >= 1.0:
        return f"${cost_usd:.2f}"
    elif cost_usd >= 0.01:
        return f"${cost_usd:.3f}"
    else:
        return f"${cost_usd:.4f}"
