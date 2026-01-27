"""
Budget tracker with LangSmith cost aggregation.

Provides:
- Query LangSmith list_runs() for monthly cost aggregation
- Cache costs (1hr TTL) to avoid excessive API calls
- Adaptive stagger calculation based on budget pace
- Three budget actions: pause, slowdown, warn
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from .paths import QUEUE_DIR
from .pricing import format_cost
from .schemas import CostCache, CostEntry

logger = logging.getLogger(__name__)

# Cache validity duration
CACHE_TTL_HOURS = 1.0

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

        self._client = None

    @property
    def client(self):
        """Lazy-load LangSmith client."""
        if self._client is None:
            from langsmith import Client

            self._client = Client()
        return self._client

    def _read_cache(self) -> CostCache:
        """Read cost cache from disk."""
        if self.cost_cache_file.exists():
            with open(self.cost_cache_file, "r") as f:
                return json.load(f)
        return {
            "version": "1.0",
            "periods": {},
            "last_sync": None,
        }

    def _write_cache(self, cache: CostCache) -> None:
        """Write cost cache to disk atomically."""
        temp_file = self.cost_cache_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(cache, f, indent=2)
        temp_file.rename(self.cost_cache_file)

    def _get_current_period(self) -> str:
        """Get current period key (project + month) for cache isolation."""
        from datetime import timezone
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        # Include project name so different projects have separate cache entries
        return f"{self.langsmith_project}:{month}"

    def _is_cache_valid(self, entry: CostEntry) -> bool:
        """Check if cache entry is still valid."""
        if not entry.get("last_aggregated"):
            return False

        from datetime import timezone
        last_aggregated = datetime.fromisoformat(entry["last_aggregated"])
        # Handle both naive and aware datetimes
        if last_aggregated.tzinfo is None:
            last_aggregated = last_aggregated.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_hours = (now - last_aggregated).total_seconds() / 3600
        return age_hours < CACHE_TTL_HOURS

    def get_current_month_cost(
        self,
        force_refresh: bool = False,
        show_progress: bool = False,
        max_runs: int = 10000,
    ) -> float:
        """Get total cost for current month.

        Uses cached data if available and valid, otherwise queries LangSmith.
        LangSmith provides total_cost directly on each run, so we just sum them.

        Args:
            force_refresh: Force refresh from LangSmith
            show_progress: Print progress while fetching
            max_runs: Maximum runs to process (safety limit)

        Returns:
            Total cost in USD for current month
        """
        period = self._get_current_period()
        cache = self._read_cache()

        # Check cache validity
        if not force_refresh and period in cache["periods"]:
            entry = cache["periods"][period]
            if self._is_cache_valid(entry):
                logger.debug(f"Using cached cost: {format_cost(entry['total_cost_usd'])}")
                return entry["total_cost_usd"]

        # Query LangSmith for current month's runs
        if show_progress:
            print("  Fetching costs from LangSmith...", end="", flush=True)

        from datetime import timezone
        start_of_month = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

        total_cost = 0.0
        token_breakdown: dict[str, int] = {}
        run_ids: list[str] = []
        run_count = 0

        try:
            # Use list_runs with is_root=True to get only top-level traces
            # LangSmith aggregates costs from all child runs into the root
            runs = self.client.list_runs(
                project_name=self.langsmith_project,
                start_time=start_of_month,
                is_root=True,
            )

            for run in runs:
                run_count += 1

                if run_count > max_runs:
                    logger.warning(f"Reached max_runs limit ({max_runs})")
                    break

                if show_progress and run_count % 100 == 0:
                    print(".", end="", flush=True)

                run_ids.append(str(run.id))

                # LangSmith provides cost directly - no need to calculate!
                if run.total_cost:
                    total_cost += float(run.total_cost)

                # Track token breakdown by run name (workflow type)
                if run.total_tokens:
                    run_type = run.name or "unknown"
                    token_breakdown[run_type] = (
                        token_breakdown.get(run_type, 0) + run.total_tokens
                    )

            if show_progress:
                print(f" done ({run_count} traces)")

        except Exception as e:
            error_msg = str(e)
            # Handle "project not found" gracefully (will be created on first run)
            if "not found" in error_msg.lower() or "404" in error_msg:
                if show_progress:
                    print(f" (project '{self.langsmith_project}' will be created on first run)")
                logger.info(f"Project {self.langsmith_project} not found - will be created on first run")
                return 0.0

            if show_progress:
                print(f" error: {e}")
            logger.warning(f"Failed to query LangSmith: {e}")
            # Return cached value if available
            if period in cache["periods"]:
                return cache["periods"][period]["total_cost_usd"]
            return 0.0

        # Update cache
        from datetime import timezone
        now = datetime.now(timezone.utc)

        cache["periods"][period] = {
            "period": period,
            "total_cost_usd": total_cost,
            "token_breakdown": token_breakdown,
            "run_count": run_count,
            "last_aggregated": now.isoformat(),
            "runs_included": run_ids[-100:],  # Keep last 100 for reference
        }
        cache["last_sync"] = now.isoformat()
        self._write_cache(cache)

        logger.info(
            f"Month-to-date cost: {format_cost(total_cost)} ({run_count} traces)"
        )
        return total_cost

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
        current_cost = self.get_current_month_cost(show_progress=show_progress)
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
        period = self._get_current_period()
        cache = self._read_cache()

        if period not in cache["periods"]:
            self.get_current_month_cost()  # Refresh cache
            cache = self._read_cache()

        if period not in cache["periods"]:
            return {}

        entry = cache["periods"][period]
        token_breakdown = entry.get("token_breakdown", {})

        # Convert tokens to approximate cost
        # This is rough since we don't track input/output separately in breakdown
        breakdown = {}
        for model, tokens in token_breakdown.items():
            from .pricing import get_model_pricing

            pricing = get_model_pricing(model)
            # Assume 70% input, 30% output tokens (rough estimate)
            input_tokens = int(tokens * 0.7)
            output_tokens = int(tokens * 0.3)
            cost = (input_tokens / 1_000_000) * pricing["input"]
            cost += (output_tokens / 1_000_000) * pricing["output"]
            breakdown[model] = cost

        return breakdown
