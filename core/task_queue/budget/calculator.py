"""Budget calculation and decision logic.

Determines budget status, action recommendations, and adaptive
stagger times based on current spending pace.
"""

import logging
from datetime import datetime
from typing import Literal

from ..pricing import format_cost
from .langsmith_provider import LangSmithCostProvider

logger = logging.getLogger(__name__)

# Budget action types
BudgetAction = Literal["pause", "slowdown", "warn", "ok"]


class BudgetCalculator:
    """Calculate budget status and make spending decisions."""

    def __init__(
        self,
        cost_provider: LangSmithCostProvider,
        monthly_budget: float,
        budget_action: str,
    ):
        """Initialize budget calculator.

        Args:
            cost_provider: Provider for fetching current costs
            monthly_budget: Monthly budget limit in USD
            budget_action: Action to take when over budget (pause/slowdown/warn)
        """
        self.cost_provider = cost_provider
        self.monthly_budget = monthly_budget
        self.budget_action = budget_action

    def get_budget_status(self, show_progress: bool = False) -> dict:
        """Get current budget status.

        Args:
            show_progress: Print progress while fetching from LangSmith

        Returns:
            Dict with:
            - current_cost: Current month's cost
            - monthly_budget: Budget limit
            - percent_used: Percentage of budget used
            - remaining: Remaining budget
            - action: Recommended action (pause/slowdown/warn/ok)
            - days_remaining: Days left in month
            - daily_budget_remaining: Daily budget based on remaining days
        """
        current_cost = self.cost_provider.get_current_month_cost(show_progress=show_progress)
        percent_used = (current_cost / self.monthly_budget) * 100 if self.monthly_budget > 0 else 0
        remaining = max(0, self.monthly_budget - current_cost)

        # Calculate days remaining in month
        from datetime import timezone
        now = datetime.now(timezone.utc)
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        days_remaining = (next_month - now).days

        # Determine action
        action: BudgetAction
        if percent_used >= 100:
            if self.budget_action == "pause":
                action = "pause"
            elif self.budget_action == "slowdown":
                action = "slowdown"
            else:
                action = "warn"
        elif percent_used >= 90:
            action = "slowdown" if self.budget_action in ("pause", "slowdown") else "warn"
        elif percent_used >= 75:
            action = "warn"
        else:
            action = "ok"

        return {
            "current_cost": current_cost,
            "monthly_budget": self.monthly_budget,
            "percent_used": percent_used,
            "remaining": remaining,
            "action": action,
            "days_remaining": days_remaining,
            "daily_budget_remaining": remaining / max(days_remaining, 1),
        }

    def should_proceed(self) -> tuple[bool, str]:
        """Check if workflow should proceed based on budget.

        Returns:
            (should_proceed, reason)
        """
        status = self.get_budget_status()

        if status["action"] == "pause":
            return False, (
                f"Budget exceeded ({format_cost(status['current_cost'])} / "
                f"{format_cost(status['monthly_budget'])})"
            )

        if status["action"] == "slowdown":
            return True, f"Budget warning: {status['percent_used']:.1f}% used"

        return True, "ok"

    def get_adaptive_stagger_hours(self, base_hours: float = 36.0) -> float:
        """Calculate adaptive stagger hours based on budget consumption.

        If under budget pace, can speed up. If over pace, slow down.

        Args:
            base_hours: Base stagger time in hours

        Returns:
            Adjusted stagger hours
        """
        status = self.get_budget_status()

        # Expected percent used based on day of month
        from datetime import timezone
        day_of_month = datetime.now(timezone.utc).day
        days_in_month = 30  # Approximation
        expected_percent = (day_of_month / days_in_month) * 100

        # Ratio of actual to expected usage
        if expected_percent > 0:
            usage_ratio = status["percent_used"] / expected_percent
        else:
            usage_ratio = 1.0

        # Adjust stagger hours based on ratio
        # Under budget (ratio < 1): can speed up (reduce stagger)
        # Over budget (ratio > 1): slow down (increase stagger)

        if usage_ratio < 0.5:
            # Way under budget - can run faster
            return base_hours * 0.5
        elif usage_ratio < 0.8:
            # Slightly under budget
            return base_hours * 0.75
        elif usage_ratio <= 1.2:
            # On track
            return base_hours
        elif usage_ratio <= 1.5:
            # Slightly over pace
            return base_hours * 1.5
        else:
            # Way over pace
            return base_hours * 2.0

    def get_cost_breakdown(self) -> dict[str, float]:
        """Get cost breakdown by model for current month.

        Returns:
            Dict mapping model name to cost in USD
        """
        period = self.cost_provider.cache_manager.get_current_period()
        cache = self.cost_provider.cache_manager.read_cache()

        if period not in cache["periods"]:
            self.cost_provider.get_current_month_cost()  # Refresh cache
            cache = self.cost_provider.cache_manager.read_cache()

        if period not in cache["periods"]:
            return {}

        entry = cache["periods"][period]
        token_breakdown = entry.get("token_breakdown", {})

        # Convert tokens to approximate cost
        # This is rough since we don't track input/output separately in breakdown
        breakdown = {}
        for model, tokens in token_breakdown.items():
            from ..pricing import get_model_pricing

            pricing = get_model_pricing(model)
            # Assume 70% input, 30% output tokens (rough estimate)
            input_tokens = int(tokens * 0.7)
            output_tokens = int(tokens * 0.3)
            cost = (input_tokens / 1_000_000) * pricing["input"]
            cost += (output_tokens / 1_000_000) * pricing["output"]
            breakdown[model] = cost

        return breakdown
