"""Budget tracking module for LLM cost management.

This module provides budget tracking and enforcement capabilities for
the task queue system. It fetches cost data from LangSmith, caches it
with 1-hour TTL, and provides budget status checks.

Main interface:
    BudgetTracker: Facade for all budget tracking operations

Components:
    - CostCacheManager: Cache persistence and validation
    - LangSmithCostProvider: Cost data from LangSmith API
    - BudgetCalculator: Budget status and decision logic
"""

from .tracker import BudgetTracker

__all__ = ["BudgetTracker"]
