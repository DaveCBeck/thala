"""Full literature review workflow.

Pipeline: lit_review → enhance (supervision+editing) → evening_reads → illustrate → spawn_publish

This is the complete academic research workflow that:
1. Discovers and processes academic papers
2. Generates an initial literature review
3. Enhances with supervision loops and editing
4. Produces an evening reads article series
5. Illustrates all articles with images
6. Spawns a publish_series task for scheduled draft creation
"""

import logging
from datetime import datetime
from typing import Any, Callable, Optional

from .base import BaseWorkflow

logger = logging.getLogger(__name__)


class LitReviewFullWorkflow(BaseWorkflow):
    """Full literature review with enhancement and article series generation."""

    @property
    def task_type(self) -> str:
        return "lit_review_full"

    @property
    def phases(self) -> list[str]:
        return [
            "lit_review",      # Academic literature review generation
            "supervision",     # Enhancement loops (theoretical depth + literature expansion)
            "editing",         # Structural editing
            "evening_reads",   # Article series generation
            "illustrate",      # Add images to all articles
            "spawn_publish",   # Create publish_series task
            "saving",          # Output to disk
            "complete",
        ]

    async def run(
        self,
        task: dict[str, Any],
        checkpoint_callback: Callable[[str], None],
        resume_from: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Run the full literature review workflow.

        Args:
            task: LitReviewTask/TopicTask with topic, research_questions, quality, etc.
            checkpoint_callback: Progress callback
            resume_from: Optional checkpoint for resumption

        Returns:
            Dict with status, lit_review, enhance, series, illustrated results
        """
        # Import workflows here to avoid circular imports
        from workflows.research.academic_lit_review import academic_lit_review
        from workflows.enhance import enhance_report
        from workflows.output.evening_reads import evening_reads_graph
        from workflows.output.illustrate import illustrate_graph

        topic = task["topic"]
        research_questions = task.get("research_questions")
        quality = task.get("quality", "standard")
        language = task.get("language", "en")
        date_range = task.get("date_range")

        logger.info(f"Starting lit_review_full workflow: {topic[:50]}...")

        errors = []
        lit_result = None
        enhance_result = None
        series_result = None
        illustrated_paths = {}
        publish_task_id = None

        # Phase 1: Literature review
        checkpoint_callback("lit_review")
        logger.info("Phase 1: Running academic literature review")

        try:
            lit_result = await academic_lit_review(
                topic=topic,
                research_questions=research_questions,
                quality=quality,
                language=language,
                date_range=date_range,
            )

            if not lit_result.get("final_review"):
                raise RuntimeError(
                    f"Literature review failed: {lit_result.get('errors', 'Unknown error')}"
                )

            logger.info(
                f"Lit review complete: {len(lit_result.get('paper_corpus', {}))} papers"
            )

        except Exception as e:
            logger.error(f"Literature review failed: {e}", exc_info=True)
            errors.append({"phase": "lit_review", "error": str(e)})
            return {"status": "failed", "errors": errors}

        # Phase 2: Enhancement (supervision + editing)
        checkpoint_callback("supervision")
        logger.info("Phase 2: Running enhancement workflow")

        try:
            # Get research questions for enhancement
            # Use generated ones from lit review if not provided
            enhance_questions = research_questions or lit_result.get("research_questions", [])
            if not enhance_questions:
                enhance_questions = [f"What are the key findings regarding {topic}?"]

            enhance_result = await enhance_report(
                report=lit_result["final_review"],
                topic=topic,
                research_questions=enhance_questions,
                quality=quality,
                loops="all",  # Run both supervision loops
                paper_corpus=lit_result.get("paper_corpus"),
                paper_summaries=lit_result.get("paper_summaries"),
                zotero_keys=lit_result.get("zotero_keys"),
                run_editing=True,
                run_fact_check=False,  # Disabled - adds latency without value
            )

            # Update checkpoint as we progress through enhancement phases
            if enhance_result.get("supervision_result"):
                checkpoint_callback("editing")

            if enhance_result.get("status") == "failed":
                logger.warning("Enhancement failed, using original lit review")
                final_report = lit_result["final_review"]
            else:
                final_report = enhance_result.get("final_report", lit_result["final_review"])

            if enhance_result.get("errors"):
                errors.extend(enhance_result["errors"])

            logger.info(
                f"Enhancement complete: status={enhance_result.get('status')}, "
                f"report_length={len(final_report)}"
            )

        except Exception as e:
            logger.error(f"Enhancement failed: {e}", exc_info=True)
            errors.append({"phase": "enhancement", "error": str(e)})
            # Fall back to original lit review
            final_report = lit_result["final_review"]

        # Phase 3: Evening reads article series
        checkpoint_callback("evening_reads")
        logger.info("Phase 3: Generating evening reads series")

        try:
            series_result = await evening_reads_graph.ainvoke({
                "input": {"literature_review": final_report}
            })

            if not series_result.get("final_outputs"):
                raise RuntimeError(
                    f"Series generation failed: {series_result.get('errors', 'Unknown error')}"
                )

            logger.info(
                f"Series complete: {len(series_result.get('final_outputs', []))} articles"
            )

        except Exception as e:
            logger.error(f"Evening reads failed: {e}", exc_info=True)
            errors.append({"phase": "evening_reads", "error": str(e)})
            series_result = {"final_outputs": []}

        # Phase 4: Illustrate articles
        checkpoint_callback("illustrate")
        logger.info("Phase 4: Illustrating articles")

        final_outputs = series_result.get("final_outputs", [])
        if final_outputs:
            try:
                illustrated_paths = await self._illustrate_articles(
                    task=task,
                    final_report=final_report,
                    final_outputs=final_outputs,
                    illustrate_graph=illustrate_graph,
                )
                logger.info(f"Illustrated {len(illustrated_paths)} articles")
            except Exception as e:
                logger.error(f"Illustration failed: {e}", exc_info=True)
                errors.append({"phase": "illustrate", "error": str(e)})

        # Phase 5: Spawn publish_series task
        checkpoint_callback("spawn_publish")
        logger.info("Phase 5: Spawning publish_series task")

        if illustrated_paths and not errors:
            try:
                publish_task_id = self._spawn_publish_task(
                    task=task,
                    lit_review_path=illustrated_paths.get("lit_review"),
                    illustrated_paths=illustrated_paths,
                    final_outputs=final_outputs,
                )
                logger.info(f"Spawned publish_series task: {publish_task_id}")
            except Exception as e:
                logger.error(f"Failed to spawn publish task: {e}", exc_info=True)
                errors.append({"phase": "spawn_publish", "error": str(e)})

        # Determine status
        if not lit_result.get("final_review"):
            status = "failed"
        elif errors:
            status = "partial"
        else:
            status = "success"

        return {
            "status": status,
            "topic": topic,
            "lit_review": lit_result,
            "enhance": enhance_result,
            "series": series_result,
            "final_report": final_report,
            "illustrated_paths": illustrated_paths,
            "publish_task_id": publish_task_id,
            "errors": errors,
        }

    async def _illustrate_articles(
        self,
        task: dict[str, Any],
        final_report: str,
        final_outputs: list[dict],
        illustrate_graph,
    ) -> dict[str, str]:
        """Illustrate all articles with images.

        Args:
            task: Task data
            final_report: Enhanced literature review
            final_outputs: Evening reads outputs
            illustrate_graph: The illustrate workflow graph

        Returns:
            Dict mapping article ID to illustrated file path
        """
        output_dir = self.get_output_dir()
        timestamp = self.generate_timestamp()
        topic_slug = self.slugify(task.get("topic", "unknown"))

        # Create illustration output directory
        illust_dir = output_dir / f"illustrated_{topic_slug}_{timestamp}"
        illust_dir.mkdir(exist_ok=True)

        illustrated_paths = {}

        # Illustrate the literature review
        try:
            lit_result = await illustrate_graph.ainvoke({
                "input": {
                    "markdown_document": final_report,
                    "title": f"Literature Review: {task.get('topic', 'Unknown')}",
                    "output_dir": str(illust_dir / "lit_review_images"),
                }
            })

            lit_path = illust_dir / "lit_review.md"
            lit_content = lit_result.get("illustrated_document", final_report)
            lit_path.write_text(lit_content)
            illustrated_paths["lit_review"] = str(lit_path)

        except Exception as e:
            logger.error(f"Failed to illustrate lit review: {e}")
            # Save unillustrated version
            lit_path = illust_dir / "lit_review.md"
            lit_path.write_text(final_report)
            illustrated_paths["lit_review"] = str(lit_path)

        # Illustrate each evening reads article
        for output in final_outputs:
            article_id = output["id"]
            try:
                article_result = await illustrate_graph.ainvoke({
                    "input": {
                        "markdown_document": output["content"],
                        "title": output["title"],
                        "output_dir": str(illust_dir / f"{article_id}_images"),
                    }
                })

                article_path = illust_dir / f"{article_id}.md"
                article_content = article_result.get("illustrated_document", output["content"])
                article_path.write_text(article_content)
                illustrated_paths[article_id] = str(article_path)

            except Exception as e:
                logger.error(f"Failed to illustrate {article_id}: {e}")
                # Save unillustrated version
                article_path = illust_dir / f"{article_id}.md"
                article_path.write_text(output["content"])
                illustrated_paths[article_id] = str(article_path)

        return illustrated_paths

    def _spawn_publish_task(
        self,
        task: dict[str, Any],
        lit_review_path: Optional[str],
        illustrated_paths: dict[str, str],
        final_outputs: list[dict],
    ) -> str:
        """Create a publish_series task with schedule.

        Args:
            task: Parent task data
            lit_review_path: Path to illustrated lit review
            illustrated_paths: Map of article ID to illustrated file path
            final_outputs: Evening reads outputs (for titles)

        Returns:
            New task ID
        """
        from ..queue_manager import TaskQueueManager

        queue = TaskQueueManager()
        base_date = queue.find_next_available_monday(task["category"])

        # Build title lookup from final_outputs
        titles = {out["id"]: out["title"] for out in final_outputs}

        # Build publish items following the schedule:
        # Day 0: Overview (everyone)
        # Day 1: Lit Review (only_paid)
        # Day 4: Deep Dive 1 (everyone)
        # Day 7: Deep Dive 2 (everyone)
        # Day 11: Deep Dive 3 (everyone)
        items = [
            {
                "id": "overview",
                "title": titles.get("overview", f"Overview: {task['topic']}"),
                "path": illustrated_paths.get("overview", ""),
                "day_offset": 0,
                "audience": "everyone",
                "published": False,
                "draft_id": None,
                "draft_url": None,
            },
            {
                "id": "lit_review",
                "title": f"Research Deep-Dive: {task['topic']}",
                "path": lit_review_path or "",
                "day_offset": 1,
                "audience": "only_paid",
                "published": False,
                "draft_id": None,
                "draft_url": None,
            },
            {
                "id": "deep_dive_1",
                "title": titles.get("deep_dive_1", f"Deep Dive 1: {task['topic']}"),
                "path": illustrated_paths.get("deep_dive_1", ""),
                "day_offset": 4,
                "audience": "everyone",
                "published": False,
                "draft_id": None,
                "draft_url": None,
            },
            {
                "id": "deep_dive_2",
                "title": titles.get("deep_dive_2", f"Deep Dive 2: {task['topic']}"),
                "path": illustrated_paths.get("deep_dive_2", ""),
                "day_offset": 7,
                "audience": "everyone",
                "published": False,
                "draft_id": None,
                "draft_url": None,
            },
            {
                "id": "deep_dive_3",
                "title": titles.get("deep_dive_3", f"Deep Dive 3: {task['topic']}"),
                "path": illustrated_paths.get("deep_dive_3", ""),
                "day_offset": 11,
                "audience": "everyone",
                "published": False,
                "draft_id": None,
                "draft_url": None,
            },
        ]

        return queue.add_task(
            task_type="publish_series",
            category=task["category"],
            priority=task["priority"],
            quality=task.get("quality", "standard"),
            base_date=base_date.isoformat(),
            items=items,
            source_task_id=task["id"],
        )

    def save_outputs(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, str]:
        """Save literature review and article series to .outputs/ directory.

        Note: Illustrated versions are already saved during the illustrate phase.
        This method saves additional metadata and non-illustrated versions.

        Args:
            task: Task data
            result: Workflow result

        Returns:
            Dict with paths to saved files
        """
        output_paths = {}

        # If we have illustrated paths, use those as the primary outputs
        illustrated_paths = result.get("illustrated_paths", {})
        if illustrated_paths:
            output_paths.update(illustrated_paths)
            logger.info(f"Using illustrated paths: {list(illustrated_paths.keys())}")

        # Save metadata summary
        output_dir = self.get_output_dir()
        timestamp = self.generate_timestamp()
        topic_slug = self.slugify(task.get("topic", "unknown"))

        # Save a summary file with all paths and publish task info
        summary_path = output_dir / f"summary_{topic_slug}_{timestamp}.json"
        import json
        summary = {
            "topic": task.get("topic"),
            "quality": task.get("quality"),
            "category": task.get("category"),
            "generated_at": datetime.now().isoformat(),
            "illustrated_paths": illustrated_paths,
            "publish_task_id": result.get("publish_task_id"),
            "status": result.get("status"),
            "errors": result.get("errors"),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        output_paths["summary"] = str(summary_path)

        # If no illustrated versions, save raw versions
        if not illustrated_paths:
            lit_result = result.get("lit_review", {})
            final_report = result.get("final_report") or lit_result.get("final_review")

            if final_report:
                lit_review_path = output_dir / f"lit_review_{topic_slug}_{timestamp}.md"
                with open(lit_review_path, "w") as f:
                    f.write(f"# Literature Review: {task.get('topic', 'Unknown')}\n\n")
                    f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                    f.write(f"*Quality: {task.get('quality', 'standard')}*\n\n")
                    if task.get("research_questions"):
                        f.write("## Research Questions\n\n")
                        for q in task["research_questions"]:
                            f.write(f"- {q}\n")
                        f.write("\n---\n\n")
                    f.write(final_report)

                output_paths["lit_review"] = str(lit_review_path)
                logger.info(f"Saved lit review to {lit_review_path}")

            # Save article series
            series_result = result.get("series", {})
            final_outputs = series_result.get("final_outputs", [])

            if final_outputs:
                series_dir = output_dir / f"series_{topic_slug}_{timestamp}"
                series_dir.mkdir(exist_ok=True)
                output_paths["series_dir"] = str(series_dir)

                for output in final_outputs:
                    article_path = series_dir / f"{output['id']}.md"
                    with open(article_path, "w") as f:
                        f.write(f"# {output['title']}\n\n")
                        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                        f.write("---\n\n")
                        f.write(output["content"])
                    output_paths[output["id"]] = str(article_path)

                logger.info(f"Saved {len(final_outputs)} articles to {series_dir}")

        return output_paths
