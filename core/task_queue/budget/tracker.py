"""Budget tracker facade for LLM cost management.

Provides a unified interface to budget tracking by composing:
- CostCacheManager: Cache persistence and validation
- LangSmithCostProvider: Cost data from LangSmith API
- BudgetCalculator: Budget status and decision logic
"""

import os
from pathlib import Path
from typing import Literal, Optional

from ..paths import QUEUE_DIR
from .cache import CostCacheManager
from .calculator import BudgetCalculator
from .langsmith_provider import LangSmithCostProvider

# Budget action types
BudgetAction = Literal["pause", "slowdown", "warn", "ok"]


class BudgetTracker:
    """Track LLM costs via LangSmith and enforce budget limits."""

    def __init__(self, queue_dir: Optional[Path] = None):
        """Initialize the budget tracker.

        Args:
            queue_dir: Override queue directory (for testing)
        """
        self.queue_dir = queue_dir or QUEUE_DIR
        self.cost_cache_file = self.queue_dir / "cost_cache.json"

        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration from environment
        self.monthly_budget = float(os.getenv("THALA_MONTHLY_BUDGET", "100.0"))
        self.budget_action = os.getenv("THALA_BUDGET_ACTION", "pause")
        # Use dedicated queue project for budget isolation from manual testing
        self.langsmith_project = os.getenv("THALA_QUEUE_PROJECT", "thala-queue")

        # Initialize components
        self.cache_manager = CostCacheManager(self.cost_cache_file, self.langsmith_project)
        self.cost_provider = LangSmithCostProvider(self.cache_manager, self.langsmith_project)
        self.calculator = BudgetCalculator(
            self.cost_provider, self.monthly_budget, self.budget_action
        )

    @property
    def client(self):
        """Lazy-load LangSmith client."""
        return self.cost_provider.client

    def get_current_month_cost(
        self,
        force_refresh: bool = False,
        show_progress: bool = False,
        max_runs: int = 10000,
    ) -> float:
        """Get total cost for current month.

        Uses cached data if available and valid, otherwise queries LangSmith.

        Args:
            force_refresh: Force refresh from LangSmith
            show_progress: Print progress while fetching
            max_runs: Maximum runs to process (safety limit)

        Returns:
            Total cost in USD for current month
        """
        return self.cost_provider.get_current_month_cost(
            force_refresh=force_refresh,
            show_progress=show_progress,
            max_runs=max_runs,
        )

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
        return self.calculator.get_budget_status(show_progress=show_progress)

    def should_proceed(self) -> tuple[bool, str]:
        """Check if workflow should proceed based on budget.

        Returns:
            (should_proceed, reason)
        """
        return self.calculator.should_proceed()

    def get_adaptive_stagger_hours(self, base_hours: float = 36.0) -> float:
        """Calculate adaptive stagger hours based on budget consumption.

        Args:
            base_hours: Base stagger time in hours

        Returns:
            Adjusted stagger hours
        """
        return self.calculator.get_adaptive_stagger_hours(base_hours=base_hours)

    def get_cost_breakdown(self) -> dict[str, float]:
        """Get cost breakdown by model for current month.

        Returns:
            Dict mapping model name to cost in USD
        """
        return self.calculator.get_cost_breakdown()
