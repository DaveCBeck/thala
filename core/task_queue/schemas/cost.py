"""Cost tracking TypedDict schemas for the task queue system."""

from typing import Any, Optional

from typing_extensions import TypedDict


class CostEntry(TypedDict):
    """Cached cost data for a time period."""

    period: str  # "2026-01" (monthly)
    total_cost_usd: float
    token_breakdown: dict[str, int]  # model_name -> tokens
    run_count: int
    last_aggregated: str  # ISO datetime
    runs_included: list[str]  # Last N run_ids for reference


class CostCache(TypedDict):
    """Cost tracking cache (cost_cache.json)."""

    version: str
    periods: dict[str, CostEntry]  # period -> CostEntry
    last_sync: Optional[str]  # Last full LangSmith sync


class IncrementalState(TypedDict):
    """Incremental checkpoint state for mid-phase resumption.

    Stored at .thala/queue/incremental/{task_id}.json.gz (gzip compressed)
    Allows resuming iterative phases (paper processing, supervision loops)
    from the last checkpoint rather than restarting the entire phase.

    Compression: Uses gzip compression (~10-30x size reduction for JSON).

    Delta-based checkpointing:
        For supervision loops, partial_results stores only delta state:
        - current_review: The current LLM output (essential for resume)
        - iteration: Current iteration count
        - new_dois_added: List of DOIs added in this iteration (not full corpus)

        On resume, full state is reconstructed by:
        1. Loading original corpus from phase_outputs["lit_result"]
        2. Re-fetching newly added papers from Elasticsearch by DOI
        3. Merging to reconstruct full state

        This reduces checkpoint size from ~10MB to ~200KB (50x reduction),
        plus gzip compression gives additional ~10x reduction.
    """

    task_id: str
    phase: str
    iteration_count: int  # Number of items processed
    checkpoint_interval: int  # Stored for debugging/observability (not used in logic)
    partial_results: dict[str, Any]  # Keyed by identifier (DOI, loop_id, etc.)
    last_checkpoint_at: str  # ISO timestamp
