"""Parallel execution command implementation."""

import asyncio

from ..parallel import run_parallel_tasks


def cmd_parallel(args):
    """Run multiple tasks concurrently."""
    print(f"Starting parallel execution: {args.count} tasks, {args.stagger}min stagger")
    results = asyncio.run(
        run_parallel_tasks(
            count=args.count,
            stagger_minutes=args.stagger,
        )
    )

    # Summary
    succeeded = sum(1 for r in results if not isinstance(r, BaseException))
    failed = len(results) - succeeded
    print(f"\nParallel execution complete: {succeeded} succeeded, {failed} failed")
