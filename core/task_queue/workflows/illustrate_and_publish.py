"""Illustrate-and-publish workflow.

Budget-aware illustration + immediate Substack draft publishing.
Spawned by lit_review_full after saving unillustrated articles to disk.

This workflow:
1. Loads the manifest written by lit_review_full
2. For each article (that isn't already done):
   a. Checks daily Imagen budget via try_acquire()
   b. Illustrates the article
   c. Publishes as a Substack draft
   d. Persists per-article progress
3. Defers with next_run_after when budget exhausted
4. Completes when all articles are illustrated and drafted
"""

from __future__ import annotations

import json
import logging

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from langsmith import traceable

if TYPE_CHECKING:
    from workflows.output.illustrate.config import IllustrateConfig

from .base import BaseWorkflow

logger = logging.getLogger(__name__)

# Hours to wait before retrying after budget exhaustion
DEFER_HOURS = 3


class IllustrateAndPublishWorkflow(BaseWorkflow):
    """Budget-aware illustration and draft publishing workflow."""

    @property
    def task_type(self) -> str:
        return "illustrate_and_publish"

    @property
    def phases(self) -> list[str]:
        return ["processing", "complete"]

    def get_task_identifier(self, task: dict[str, Any]) -> str:
        source_id = task.get("source_task_id", "unknown")[:8]
        topic = task.get("topic", "unknown")[:40]
        return f"illustrate({source_id}): {topic}"

    @traceable(run_type="chain", name="Task_IllustrateAndPublish")
    async def run(
        self,
        task: dict[str, Any],
        checkpoint_callback: Callable[[str], None],
        resume_from: Optional[dict] = None,
        *,
        flush_checkpoints: Optional[Callable[[], Awaitable[None]]] = None,
        update_items_callback: Optional[Callable[[str, list[dict]], Awaitable[None]]] = None,
    ) -> dict[str, Any]:
        """Run the illustrate-and-publish workflow.

        Args:
            task: IllustrateAndPublishTask with items, manifest_path, etc.
            checkpoint_callback: Progress callback
            resume_from: Optional checkpoint for resumption
            flush_checkpoints: Async function to await pending checkpoint writes
            update_items_callback: Async callback to persist item updates to queue

        Returns:
            Dict with status (success/deferred/failed), plus details
        """
        from core.task_queue.rate_limits import get_imagen_daily_tracker
        from workflows.output.illustrate import illustrate_graph
        from workflows.output.illustrate.config import IllustrateConfig

        checkpoint_callback("processing")

        items = task["items"]
        manifest_path = task.get("manifest_path")
        category = task.get("category", "")

        # Load manifest for metadata
        manifest = self._load_manifest(manifest_path)
        if not manifest:
            return {"status": "failed", "errors": [{"phase": "processing", "error": "Manifest not found or invalid"}]}

        output_dir = Path(manifest["output_dir"])

        # Fast-fail: check if any budget remains
        daily_tracker = get_imagen_daily_tracker()
        unillustrated = [i for i in items if not i.get("illustrated")]
        if unillustrated:
            remaining = await daily_tracker.remaining()
            if remaining == 0:
                next_run = (datetime.now(timezone.utc) + timedelta(hours=DEFER_HOURS)).isoformat()
                logger.info(f"No daily budget remaining, deferring {len(unillustrated)} articles")
                return {"status": "deferred", "next_run_after": next_run}

        # Select config factory based on task quality
        quality = task.get("quality", "standard")
        _config_factories = {
            "quick": IllustrateConfig.quick,
            "quality": IllustrateConfig.quality,
        }
        config_factory = _config_factories.get(quality)

        # Per-article loop: illustrate → publish → checkpoint
        errors = []
        progress_made = False
        cached_vi = None

        for item in items:
            # Skip fully completed items
            if item.get("draft_id"):
                continue

            # Illustrate if not yet done
            if not item.get("illustrated"):
                # Non-consuming check only — the actual try_acquire() happens
                # inside generate_article_header() (image_utils.py) which is
                # the single source of truth for daily budget consumption.
                if await daily_tracker.remaining() < 1:
                    logger.info("Daily budget exhausted after illustrating some articles")
                    break

                try:
                    vi_kwarg = {"visual_identity_override": cached_vi} if cached_vi else {}
                    config = config_factory(**vi_kwarg) if config_factory else (
                        IllustrateConfig(**vi_kwarg) if vi_kwarg else None
                    )
                    illustrated_path, article_result = await self._illustrate_article(
                        item,
                        output_dir,
                        illustrate_graph,
                        config=config,
                    )
                    item["illustrated"] = True
                    item["illustrated_path"] = str(illustrated_path)
                    progress_made = True

                    if cached_vi is None:
                        vi = article_result.get("visual_identity")
                        if vi:
                            cached_vi = vi
                            logger.info(f"Cached visual identity: style='{vi.primary_style}'")
                except Exception as e:
                    logger.error(f"Failed to illustrate {item['id']}: {e}")
                    errors.append({"item": item["id"], "error": str(e)})
                    # Save unillustrated version as fallback
                    source = Path(item["source_path"])
                    if source.exists():
                        fallback_path = output_dir / f"{item['id']}_unillustrated.md"
                        fallback_path.write_text(source.read_text())
                        item["illustrated"] = True
                        item["illustrated_path"] = str(fallback_path)
                        progress_made = True

            # Publish (whether just illustrated or previously illustrated but not drafted)
            if item.get("illustrated") and not item.get("draft_id"):
                try:
                    draft_result = await self._publish_draft(item, category)
                    item["draft_id"] = draft_result.get("post_id")
                    item["draft_url"] = draft_result.get("draft_url")
                    progress_made = True
                except Exception as e:
                    logger.error(f"Failed to publish {item['id']}: {e}")
                    errors.append({"item": item["id"], "error": str(e)})

            # Persist progress after each article
            if flush_checkpoints:
                await flush_checkpoints()
            if update_items_callback:
                await update_items_callback(task["id"], items)
            checkpoint_callback("processing", phase_outputs={"last_item": item["id"]})

        # Status determination
        all_done = all(i.get("draft_id") for i in items)
        if all_done:
            return {"status": "success", "items": items}

        remaining_items = [i for i in items if not i.get("draft_id")]
        if progress_made or not errors:
            # Still work to do — defer for later
            next_run = (datetime.now(timezone.utc) + timedelta(hours=DEFER_HOURS)).isoformat()
            logger.info(f"Deferring {len(remaining_items)} remaining articles until {next_run}")
            return {"status": "deferred", "next_run_after": next_run, "items": items}

        return {"status": "failed", "errors": errors, "items": items}

    def save_outputs(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, str]:
        """No additional outputs to save — articles saved during processing."""
        return {}

    def _load_manifest(self, manifest_path: str | None) -> dict | None:
        """Load and validate manifest.json."""
        if not manifest_path:
            return None
        path = Path(manifest_path)
        if not path.exists():
            logger.error(f"Manifest not found: {manifest_path}")
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load manifest: {e}")
            return None

    async def _illustrate_article(
        self,
        item: dict,
        output_dir: Path,
        illustrate_graph: Any,
        config: IllustrateConfig | None = None,
    ) -> tuple[Path, dict]:
        """Illustrate a single article and save to disk.

        Args:
            item: Article metadata dict with source_path, title, id.
            output_dir: Directory to write illustrated output into.
            illustrate_graph: LangGraph CompiledGraph for illustration pipeline.
            config: Optional illustration config (carries visual_identity_override).

        Returns:
            Tuple of (illustrated_path, graph_result_dict).
        """
        from workflows.output.illustrate.config import IllustrateConfig as _IC

        source_path = Path(item["source_path"])
        content = source_path.read_text()

        article_result = await illustrate_graph.ainvoke(
            {
                "input": {
                    "markdown_document": content,
                    "title": item["title"],
                    "output_dir": str(output_dir / f"{item['id']}_images"),
                },
                "config": config or _IC(),
            }
        )

        illustrated_content = article_result.get("illustrated_document", content)
        illustrated_path = output_dir / f"{item['id']}_illustrated.md"
        illustrated_path.write_text(illustrated_content)

        logger.info(f"Illustrated article: {item['title'][:50]}")
        return illustrated_path, article_result

    async def _publish_draft(self, item: dict, category: str) -> dict:
        """Publish an illustrated article as a Substack draft."""
        import asyncio

        from core.task_queue.paths import SUBSTACK_COOKIES_FILE
        from core.task_queue.workflows.shared.publication_config import load_publication_config
        from utils.substack_publish import SubstackConfig, SubstackPublisher

        # Load publication config for this category
        pub_config = load_publication_config(category)

        # Read illustrated content
        content_path = Path(item.get("illustrated_path", item["source_path"]))
        content = content_path.read_text()

        # Determine audience: overview is "everyone", others could vary
        audience = "everyone"
        if item["id"] == "lit_review":
            audience = "only_paid"

        config = SubstackConfig(
            cookies_path=str(SUBSTACK_COOKIES_FILE),
            publication_url=pub_config.get("publication_url"),
            audience=audience,
        )
        publisher = SubstackPublisher(config)

        # SubstackPublisher.create_draft is sync — run in thread
        result = await asyncio.to_thread(
            publisher.create_draft,
            markdown=content,
            title=item["title"],
        )

        if not result.get("success"):
            raise RuntimeError(f"Draft creation failed: {result.get('error', 'unknown')}")

        logger.info(f"Published draft: {item['title'][:50]} -> {result.get('draft_url')}")
        return result
