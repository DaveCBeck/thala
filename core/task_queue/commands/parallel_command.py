"""Parallel execution command implementation."""

import asyncio
import os
import sys

from ..parallel import run_parallel_tasks


def cmd_parallel(args):
    """Run multiple tasks concurrently."""
    if args.count < 1:
        print("Error: --count must be >= 1")
        sys.exit(1)
    if args.stagger < 0:
        print("Error: --stagger must be >= 0")
        sys.exit(1)

    # Auto-scale broker concurrency for parallel execution if not already configured
    if "THALA_LLM_BROKER_MAX_CONCURRENT_SYNC" not in os.environ:
        recommended = args.count * 3
        os.environ["THALA_LLM_BROKER_MAX_CONCURRENT_SYNC"] = str(recommended)
        print(f"Auto-scaling broker concurrency to {recommended} (count * 3)")

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
