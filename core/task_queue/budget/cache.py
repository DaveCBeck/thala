"""Cost cache management for budget tracking.

Handles reading, writing, and validation of cost cache data with
1-hour TTL to reduce LangSmith API calls.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from ..schemas.cost import CostCache, CostEntry

logger = logging.getLogger(__name__)

# Cache validity duration
CACHE_TTL_HOURS = 1.0


class CostCacheManager:
    """Manage cost cache persistence and validation."""

    def __init__(self, cache_file: Path, langsmith_project: str):
        """Initialize cache manager.

        Args:
            cache_file: Path to cost_cache.json
            langsmith_project: LangSmith project name for cache isolation
        """
        self.cache_file = cache_file
        self.langsmith_project = langsmith_project

    def read_cache(self) -> CostCache:
        """Read cost cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {
            "version": "1.0",
            "periods": {},
            "last_sync": None,
        }

    def write_cache(self, cache: CostCache) -> None:
        """Write cost cache to disk atomically."""
        temp_file = self.cache_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(cache, f, indent=2)
        try:
            temp_file.rename(self.cache_file)
        except FileNotFoundError:
            # Temp file may have been deleted by concurrent cleanup
            logger.warning(f"Temp file {temp_file} disappeared before rename - retrying write")
            with open(self.cache_file, "w") as f:
                json.dump(cache, f, indent=2)

    def get_current_period(self) -> str:
        """Get current period key (project + month) for cache isolation."""
        from datetime import timezone
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        # Include project name so different projects have separate cache entries
        return f"{self.langsmith_project}:{month}"

    def is_cache_valid(self, entry: CostEntry) -> bool:
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
