#!/usr/bin/env python3
"""
Setup Elasticsearch index templates.

Creates index templates with proper mappings for all stores.
Run this once when setting up ES, or after schema changes.

Usage:
    python -m core.stores.setup_indices
    python -m core.stores.setup_indices --reset  # Delete and recreate indices
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from elasticsearch import AsyncElasticsearch

from core.config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Index routing: which ES instance hosts which indices
ES_COHERENCE_HOST = "http://localhost:9201"
ES_FORGOTTEN_HOST = "http://localhost:9200"

COHERENCE_INDICES = ["store_l0", "store_l1", "store_l2", "coherence"]
FORGOTTEN_INDICES = ["who_i_was", "forgotten"]


async def apply_template(client: AsyncElasticsearch, name: str) -> None:
    """Apply an index template from JSON file."""
    template_file = TEMPLATES_DIR / f"{name}.json"
    if not template_file.exists():
        logger.error(f"Template file not found: {template_file}")
        return

    with open(template_file) as f:
        template = json.load(f)

    await client.indices.put_index_template(name=name, body=template)
    logger.info(f"Applied template: {name}")


async def create_index(client: AsyncElasticsearch, name: str) -> None:
    """Create an index if it doesn't exist."""
    exists = await client.indices.exists(index=name)
    if not exists:
        await client.indices.create(index=name)
        logger.info(f"Created index: {name}")
    else:
        logger.info(f"Index already exists: {name}")


async def delete_index(client: AsyncElasticsearch, name: str) -> None:
    """Delete an index if it exists."""
    exists = await client.indices.exists(index=name)
    if exists:
        await client.indices.delete(index=name)
        logger.info(f"Deleted index: {name}")


async def setup_indices(reset: bool = False) -> None:
    """Set up all index templates and create indices."""
    coherence_client = AsyncElasticsearch(hosts=[ES_COHERENCE_HOST])
    forgotten_client = AsyncElasticsearch(hosts=[ES_FORGOTTEN_HOST])

    try:
        # Apply templates and create indices on coherence instance
        logger.info(f"Setting up coherence instance ({ES_COHERENCE_HOST})...")
        for name in COHERENCE_INDICES:
            if reset:
                await delete_index(coherence_client, name)
            await apply_template(coherence_client, name)
            await create_index(coherence_client, name)

        # Apply templates and create indices on forgotten instance
        logger.info(f"Setting up forgotten instance ({ES_FORGOTTEN_HOST})...")
        for name in FORGOTTEN_INDICES:
            if reset:
                await delete_index(forgotten_client, name)
            await apply_template(forgotten_client, name)
            await create_index(forgotten_client, name)

        logger.info("Index setup complete!")

    finally:
        await coherence_client.close()
        await forgotten_client.close()


async def show_status() -> None:
    """Show current index status."""
    coherence_client = AsyncElasticsearch(hosts=[ES_COHERENCE_HOST])
    forgotten_client = AsyncElasticsearch(hosts=[ES_FORGOTTEN_HOST])

    try:
        print("\n=== ES Coherence (9201) ===")
        for name in COHERENCE_INDICES:
            exists = await coherence_client.indices.exists(index=name)
            if exists:
                stats = await coherence_client.indices.stats(index=name)
                doc_count = stats["indices"][name]["primaries"]["docs"]["count"]
                print(f"  {name}: {doc_count} docs")
            else:
                print(f"  {name}: (not created)")

        print("\n=== ES Forgotten (9200) ===")
        for name in FORGOTTEN_INDICES:
            exists = await forgotten_client.indices.exists(index=name)
            if exists:
                stats = await forgotten_client.indices.stats(index=name)
                doc_count = stats["indices"][name]["primaries"]["docs"]["count"]
                print(f"  {name}: {doc_count} docs")
            else:
                print(f"  {name}: (not created)")

    finally:
        await coherence_client.close()
        await forgotten_client.close()


def main():
    parser = argparse.ArgumentParser(description="Setup Elasticsearch indices")
    parser.add_argument(
        "--reset", action="store_true", help="Delete and recreate indices"
    )
    parser.add_argument("--status", action="store_true", help="Show index status")
    args = parser.parse_args()

    if args.status:
        asyncio.run(show_status())
    else:
        asyncio.run(setup_indices(reset=args.reset))


if __name__ == "__main__":
    main()
