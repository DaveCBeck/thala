"""Budget tracker - backward compatibility shim.

This module re-exports BudgetTracker from the budget package for
backward compatibility with existing imports.

New code should import from:
    from core.task_queue.budget import BudgetTracker
"""

from .budget import BudgetTracker

__all__ = ["BudgetTracker"]
