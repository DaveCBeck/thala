"""LangSmith cost data provider.

Fetches monthly cost data from LangSmith API and aggregates costs
from trace runs. Handles caching and graceful error handling.
"""

import logging
from datetime import datetime

from ..pricing import format_cost
from .cache import CostCacheManager

logger = logging.getLogger(__name__)


class LangSmithCostProvider:
    """Fetch and aggregate costs from LangSmith API."""

    def __init__(self, cache_manager: CostCacheManager, langsmith_project: str):
        """Initialize LangSmith cost provider.

        Args:
            cache_manager: Cache manager for storing/retrieving cost data
            langsmith_project: LangSmith project name to query
        """
        self.cache_manager = cache_manager
        self.langsmith_project = langsmith_project
        self._client = None

    @property
    def client(self):
        """Lazy-load LangSmith client."""
        if self._client is None:
            from langsmith import Client

            self._client = Client()
        return self._client

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
        period = self.cache_manager.get_current_period()
        cache = self.cache_manager.read_cache()

        # Check cache validity
        if not force_refresh and period in cache["periods"]:
            entry = cache["periods"][period]
            if self.cache_manager.is_cache_valid(entry):
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
        self.cache_manager.write_cache(cache)

        logger.info(
            f"Month-to-date cost: {format_cost(total_cost)} ({run_count} traces)"
        )
        return total_cost
