"""Web research workflow.

Pipeline: deep_research → evening_reads

This workflow performs web-based research (not academic papers) and
produces an evening reads article series from the findings.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Optional

from .base import BaseWorkflow

logger = logging.getLogger(__name__)


class WebResearchWorkflow(BaseWorkflow):
    """Web research with article series generation."""

    @property
    def task_type(self) -> str:
        return "web_research"

    @property
    def phases(self) -> list[str]:
        return [
            "research",        # Deep web research
            "evening_reads",   # Article series generation
            "saving",          # Output to disk
            "complete",
        ]

    def _get_completed_phases(self, checkpoint: dict) -> set[str]:
        """Get phases completed before the checkpoint phase."""
        current_phase = checkpoint.get("phase", "")
        try:
            current_idx = self.phases.index(current_phase)
            return set(self.phases[:current_idx])
        except ValueError:
            return set()

    async def run(
        self,
        task: dict[str, Any],
        checkpoint_callback: Callable[[str], None],
        resume_from: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Run the web research workflow.

        Args:
            task: WebResearchTask with query, quality, language, etc.
            checkpoint_callback: Progress callback
            resume_from: Optional checkpoint for resumption

        Returns:
            Dict with status, research, series results
        """
        # Import workflows here to avoid circular imports
        from workflows.research.web_research import deep_research
        from workflows.output.evening_reads import evening_reads_graph

        query = task.get("query") or task.get("topic")  # Support both field names
        quality = task.get("quality", "standard")
        language = task.get("language")

        # Determine which phases to skip if resuming
        completed_phases: set[str] = set()
        phase_outputs: dict[str, Any] = {}

        if resume_from:
            completed_phases = self._get_completed_phases(resume_from)
            phase_outputs = resume_from.get("phase_outputs", {})
            logger.info(
                f"Resuming web_research from {resume_from['phase']}, "
                f"skipping completed phases: {completed_phases}"
            )
        else:
            logger.info(f"Starting web_research workflow: {query[:50]}...")

        errors = []
        research_result = None
        series_result = None

        # Phase 1: Deep research
        if "research" in completed_phases:
            research_result = phase_outputs.get("research_result")
            logger.info("Skipping research phase (already complete)")
        else:
            checkpoint_callback("research")
            logger.info("Phase 1: Running deep web research")

            try:
                research_result = await deep_research(
                    query=query,
                    quality=quality,
                    language=language,
                )

                if not research_result.get("final_report"):
                    raise RuntimeError(
                        f"Web research failed: {research_result.get('errors', 'Unknown error')}"
                    )

                logger.info(
                    f"Research complete: {research_result.get('source_count', 0)} sources, "
                    f"status={research_result.get('status')}"
                )

                if research_result.get("errors"):
                    errors.extend([
                        {"phase": "research", **err} if isinstance(err, dict) else {"phase": "research", "error": str(err)}
                        for err in research_result["errors"]
                    ])

                # Save outputs for potential resume
                checkpoint_callback("research", phase_outputs={"research_result": research_result})

            except Exception as e:
                logger.error(f"Web research failed: {e}", exc_info=True)
                errors.append({"phase": "research", "error": str(e)})
                return {"status": "failed", "errors": errors}

        # Phase 2: Evening reads article series
        if "evening_reads" in completed_phases:
            series_result = phase_outputs.get("series_result")
            logger.info("Skipping evening_reads phase (already complete)")
        else:
            checkpoint_callback("evening_reads")
            logger.info("Phase 2: Generating evening reads series")

            # Load editorial stance for the publication
            from workflows.output.evening_reads.editorial import load_editorial_stance
            editorial_stance = load_editorial_stance(task.get("category", ""))
            if editorial_stance:
                logger.info(f"Using editorial stance for category: {task.get('category')}")

            try:
                series_result = await evening_reads_graph.ainvoke({
                    "input": {
                        "literature_review": research_result["final_report"],
                        "editorial_stance": editorial_stance,
                    }
                })

                if not series_result.get("final_outputs"):
                    raise RuntimeError(
                        f"Series generation failed: {series_result.get('errors', 'Unknown error')}"
                    )

                logger.info(
                    f"Series complete: {len(series_result.get('final_outputs', []))} articles"
                )

                # Save outputs for potential resume
                checkpoint_callback(
                    "evening_reads",
                    phase_outputs={
                        "research_result": research_result,
                        "series_result": series_result,
                    },
                )

            except Exception as e:
                logger.error(f"Evening reads failed: {e}", exc_info=True)
                errors.append({"phase": "evening_reads", "error": str(e)})
                series_result = {"final_outputs": []}

        # Determine status
        if not research_result or not research_result.get("final_report"):
            status = "failed"
        elif errors:
            status = "partial"
        else:
            status = "success"

        return {
            "status": status,
            "query": query,
            "research": research_result,
            "series": series_result,
            "final_report": research_result.get("final_report") if research_result else None,
            "errors": errors,
        }

    def save_outputs(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, str]:
        """Save research report and article series to .thala/output/ directory.

        Args:
            task: Task data
            result: Workflow result

        Returns:
            Dict with paths to saved files
        """
        output_dir = self.get_output_dir()
        timestamp = self.generate_timestamp()
        query = task.get("query") or task.get("topic", "unknown")
        query_slug = self.slugify(query)

        output_paths = {}

        # Save research report
        research_result = result.get("research", {})
        final_report = result.get("final_report") or research_result.get("final_report")

        if final_report:
            report_path = output_dir / f"web_research_{query_slug}_{timestamp}.md"
            with open(report_path, "w") as f:
                f.write(f"# Web Research: {query}\n\n")
                f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write(f"*Quality: {task.get('quality', 'standard')}*\n\n")
                f.write(f"*Sources: {research_result.get('source_count', 'unknown')}*\n\n")
                f.write("---\n\n")
                f.write(final_report)

            output_paths["research_report"] = str(report_path)
            logger.info(f"Saved research report to {report_path}")

        # Save article series
        series_result = result.get("series", {})
        final_outputs = series_result.get("final_outputs", [])

        if final_outputs:
            series_dir = output_dir / f"series_{query_slug}_{timestamp}"
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
