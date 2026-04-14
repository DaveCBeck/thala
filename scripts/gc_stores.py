#!/usr/bin/env python3
"""CLI wrapper for core.stores.gc.

Walks a folder of markdown, extracts [@KEY] citations, and GCs Zotero + ES +
Chroma against that keep-set.

Usage:
    .venv/bin/python scripts/gc_stores.py                  # dry-run, source=incoming
    .venv/bin/python scripts/gc_stores.py --execute        # actually delete
    .venv/bin/python scripts/gc_stores.py --source path/   # different source folder
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stores.chroma import ChromaStore  # noqa: E402
from core.stores.elasticsearch.client import ElasticsearchStores  # noqa: E402
from core.stores.gc import (  # noqa: E402
    GCPlan,
    Inventory,
    build_inventory,
    build_plan,
    collect_cited_keys_from_dir,
    execute_plan,
)
from core.stores.zotero import ZoteroStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("gc_stores")
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

DELETE_REASON = "GC: not cited by documents in incoming/"


def report(plan: GCPlan, inv: Inventory, file_count: int) -> None:
    print()
    print("=" * 72)
    print(f"Scanned {file_count} markdown files; found {len(plan.cited_keys)} unique citation keys.")
    print("=" * 72)

    def section(label: str, kept: int, dropped: int, total: int) -> None:
        pct = (dropped / total * 100) if total else 0
        print(f"{label:<20} total={total:<6} keep={kept:<6} drop={dropped:<6} ({pct:.1f}% drop)")

    section("Zotero items", len(plan.keep_zotero), len(plan.drop_zotero), len(inv.zotero_all))
    section("ES store_l0 (ext)", len(plan.keep_l0), len(plan.drop_l0), len(inv.l0_by_zkey))
    print(f"{'ES store_l0 (int)':<20} preserved={plan.l0_internal}")
    section("ES store_l1", len(inv.l1) - len(plan.drop_l1), len(plan.drop_l1), len(inv.l1))
    section("ES store_l2", len(inv.l2) - len(plan.drop_l2), len(plan.drop_l2), len(inv.l2))
    section("Chroma (linked)", 0, len(plan.drop_chroma),
            sum(1 for _, z in inv.chroma_rows if z))
    chroma_unlinked = sum(1 for _, z in inv.chroma_rows if not z)
    if chroma_unlinked:
        print(f"{'Chroma (unlinked)':<20} preserved={chroma_unlinked}")

    if plan.l0_uncited_zkeys:
        print()
        print(f"Note: {len(plan.l0_uncited_zkeys)} citation keys have no L0 ES record "
              f"(cited but never ingested or already gone).")
        for k in sorted(list(plan.l0_uncited_zkeys))[:10]:
            print(f"  - {k}")
        if len(plan.l0_uncited_zkeys) > 10:
            print(f"  ... and {len(plan.l0_uncited_zkeys) - 10} more")

    cited_not_in_zotero = plan.cited_keys - set(inv.zotero_all.keys())
    if cited_not_in_zotero:
        print()
        print(f"Warning: {len(cited_not_in_zotero)} citation keys missing from Zotero library:")
        for k in sorted(list(cited_not_in_zotero))[:10]:
            print(f"  - {k}")

    print()


async def run(source: Path, do_execute: bool) -> None:
    logger.info(f"Collecting citations from {source}...")
    cited, files = collect_cited_keys_from_dir(source)
    logger.info(f"Found {len(cited)} unique keys across {len(files)} files.")
    if not cited:
        logger.error("No citation keys found; aborting to avoid wiping everything.")
        return

    zotero = ZoteroStore()
    try:
        async with ElasticsearchStores() as es:
            chroma = ChromaStore(es_stores=es)

            logger.info("Loading inventories (Zotero, ES, Chroma)...")
            inv = await build_inventory(zotero, es, chroma)
            plan = build_plan(cited, inv)
            report(plan, inv, len(files))

            if not do_execute:
                print("Dry-run only. Pass --execute to perform deletions.")
                return

            if plan.total_dropped == 0:
                logger.info("Nothing to delete.")
                return

            confirm = input("Type 'DELETE' to proceed: ").strip()
            if confirm != "DELETE":
                logger.info("Aborted.")
                return

            await execute_plan(plan, zotero, es, chroma, reason=DELETE_REASON)
            logger.info("Done.")
    finally:
        await zotero.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).parent.parent / "incoming",
        help="Folder containing markdown documents that define the keep-set.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform deletions. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        logger.error(f"Source folder does not exist: {args.source}")
        sys.exit(1)

    asyncio.run(run(args.source, args.execute))


if __name__ == "__main__":
    main()
