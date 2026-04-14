"""Garbage collection for Zotero, Elasticsearch, and Chroma.

Given a set of cited Zotero keys, partitions every external record in the
stores into keep/drop and (optionally) performs the deletions. Deletes flow
through the existing archival paths (forgotten / who_i_was).

Single-hop dependency rule: an L1/L2 record is dropped iff any of its
source_ids points at a dropped L0. Internal (non-Zotero-linked) records are
never touched.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from elasticsearch.helpers import async_scan

from .chroma import ChromaStore
from .elasticsearch.client import ElasticsearchStores
from .zotero import ZoteroStore

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[@([A-Za-z0-9]{8})\]")
DEFAULT_CONCURRENCY = 10


def collect_cited_keys_from_files(files: list[Path]) -> set[str]:
    """Extract unique [@KEY] citations from a list of markdown files."""
    keys: set[str] = set()
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        keys.update(CITATION_RE.findall(text))
    return keys


def collect_cited_keys_from_dir(
    source: Path, pattern: str = "*.md"
) -> tuple[set[str], list[Path]]:
    """Walk a directory for files matching pattern, extract citation keys."""
    files = list(source.rglob(pattern))
    return collect_cited_keys_from_files(files), files


@dataclass
class Inventory:
    zotero_all: dict[str, str] = field(default_factory=dict)
    l0_by_zkey: dict[str, UUID] = field(default_factory=dict)
    l0_internal: int = 0
    l1: list[tuple[UUID, list[UUID]]] = field(default_factory=list)
    l2: list[tuple[UUID, list[UUID]]] = field(default_factory=list)
    chroma_rows: list[tuple[UUID, str | None]] = field(default_factory=list)


@dataclass
class GCPlan:
    cited_keys: set[str]
    keep_zotero: set[str]
    drop_zotero: set[str]
    keep_l0: dict[str, UUID]
    drop_l0: dict[str, UUID]
    drop_l1: list[UUID]
    drop_l2: list[UUID]
    drop_chroma: list[UUID]
    l0_internal: int
    l0_uncited_zkeys: set[str]

    @property
    def total_dropped(self) -> int:
        return (
            len(self.drop_zotero)
            + len(self.drop_l0)
            + len(self.drop_l1)
            + len(self.drop_l2)
            + len(self.drop_chroma)
        )


# ---------- inventory loaders ----------


async def _load_zotero(zotero: ZoteroStore) -> dict[str, str]:
    items = await zotero.get_all(limit=50000)
    return {it.key: (it.title or "") for it in items}


async def _load_es(es: ElasticsearchStores) -> tuple[dict[str, UUID], int, list, list]:
    l0_by_zkey: dict[str, UUID] = {}
    internal = 0
    l1: list[tuple[UUID, list[UUID]]] = []
    l2: list[tuple[UUID, list[UUID]]] = []

    client = es._coherence_client

    async for hit in async_scan(
        client,
        index="store_l0",
        query={"query": {"match_all": {}}},
        _source=["id", "zotero_key", "source_type"],
        size=1000,
    ):
        src = hit["_source"]
        rid = UUID(src["id"])
        if src.get("source_type") == "external" and src.get("zotero_key"):
            l0_by_zkey[src["zotero_key"]] = rid
        else:
            internal += 1

    for level, bucket in ((1, l1), (2, l2)):
        async for hit in async_scan(
            client,
            index=f"store_l{level}",
            query={"query": {"match_all": {}}},
            _source=["id", "source_ids"],
            size=1000,
        ):
            src = hit["_source"]
            rid = UUID(src["id"])
            source_ids = [UUID(x) for x in (src.get("source_ids") or [])]
            bucket.append((rid, source_ids))

    return l0_by_zkey, internal, l1, l2


async def _load_chroma(chroma: ChromaStore) -> list[tuple[UUID, str | None]]:
    collection = await chroma._get_collection()
    result = await asyncio.to_thread(collection.get, include=["metadatas"])
    rows: list[tuple[UUID, str | None]] = []
    for rid, meta in zip(result["ids"], result["metadatas"] or []):
        zkey = (meta or {}).get("zotero_key")
        rows.append((UUID(rid), zkey))
    return rows


async def build_inventory(
    zotero: ZoteroStore, es: ElasticsearchStores, chroma: ChromaStore
) -> Inventory:
    zotero_all, es_tuple, chroma_rows = await asyncio.gather(
        _load_zotero(zotero),
        _load_es(es),
        _load_chroma(chroma),
    )
    l0_by_zkey, l0_internal, l1, l2 = es_tuple
    return Inventory(
        zotero_all=zotero_all,
        l0_by_zkey=l0_by_zkey,
        l0_internal=l0_internal,
        l1=l1,
        l2=l2,
        chroma_rows=chroma_rows,
    )


def build_plan(cited: set[str], inv: Inventory) -> GCPlan:
    zotero_keys = set(inv.zotero_all.keys())
    keep_zotero = cited & zotero_keys
    drop_zotero = zotero_keys - cited

    keep_l0 = {k: v for k, v in inv.l0_by_zkey.items() if k in cited}
    drop_l0 = {k: v for k, v in inv.l0_by_zkey.items() if k not in cited}
    dropped_l0_ids = set(drop_l0.values())

    drop_l1 = [rid for rid, srcs in inv.l1 if any(s in dropped_l0_ids for s in srcs)]
    drop_l2 = [rid for rid, srcs in inv.l2 if any(s in dropped_l0_ids for s in srcs)]
    drop_chroma = [rid for rid, zkey in inv.chroma_rows if zkey and zkey not in cited]

    return GCPlan(
        cited_keys=cited,
        keep_zotero=keep_zotero,
        drop_zotero=drop_zotero,
        keep_l0=keep_l0,
        drop_l0=drop_l0,
        drop_l1=drop_l1,
        drop_l2=drop_l2,
        drop_chroma=drop_chroma,
        l0_internal=inv.l0_internal,
        l0_uncited_zkeys=cited - set(inv.l0_by_zkey.keys()),
    )


async def execute_plan(
    plan: GCPlan,
    zotero: ZoteroStore,
    es: ElasticsearchStores,
    chroma: ChromaStore,
    *,
    reason: str,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(coro):
        async with sem:
            try:
                return await coro
            except Exception as e:
                logger.warning(f"delete failed: {e}")
                return None

    # Compressed layers first (they reference L0 via source_ids).
    logger.info(f"GC: deleting {len(plan.drop_l1)} L1 records")
    await asyncio.gather(*[
        guarded(es.store.delete(rid, reason=reason, compression_level=1))
        for rid in plan.drop_l1
    ])

    logger.info(f"GC: deleting {len(plan.drop_l2)} L2 records")
    await asyncio.gather(*[
        guarded(es.store.delete(rid, reason=reason, compression_level=2))
        for rid in plan.drop_l2
    ])

    logger.info(f"GC: deleting {len(plan.drop_l0)} L0 records")
    await asyncio.gather(*[
        guarded(es.store.delete(rid, reason=reason, compression_level=0))
        for rid in plan.drop_l0.values()
    ])

    logger.info(f"GC: deleting {len(plan.drop_chroma)} Chroma records")
    await asyncio.gather(*[
        guarded(chroma.delete(rid, reason=reason))
        for rid in plan.drop_chroma
    ])

    logger.info(f"GC: deleting {len(plan.drop_zotero)} Zotero items")
    await asyncio.gather(*[
        guarded(zotero.delete(key)) for key in plan.drop_zotero
    ])


async def garbage_collect(
    cited_keys: set[str],
    zotero: ZoteroStore,
    es: ElasticsearchStores,
    chroma: ChromaStore,
    *,
    reason: str,
    execute: bool = False,
) -> GCPlan:
    """Build a GC plan and optionally execute it.

    Args:
        cited_keys: Zotero keys to preserve (everything else is eligible for deletion).
        zotero, es, chroma: Store clients.
        reason: Archival reason recorded on the forgotten/who_i_was records.
        execute: When True, perform the deletions; otherwise return the plan only.
    """
    inv = await build_inventory(zotero, es, chroma)
    plan = build_plan(cited_keys, inv)
    if execute:
        await execute_plan(plan, zotero, es, chroma, reason=reason)
    return plan
