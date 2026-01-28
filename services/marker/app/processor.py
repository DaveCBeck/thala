"""Marker document processor wrapper."""

import gc
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.renderers.markdown import MarkdownRenderer
from marker.renderers.chunk import ChunkRenderer

from app.config import QUALITY_PRESETS, get_settings

logger = logging.getLogger(__name__)

# Idle timeout before unloading models (30 minutes)
MODEL_IDLE_TIMEOUT_SEC = 30 * 60


class MarkerProcessor:
    """Wrapper around Marker PDF converter with quality presets."""

    def __init__(self):
        self.settings = get_settings()
        self._models = None
        self._last_used = None
        self._lock = threading.Lock()
        self._unload_timer = None

    def _get_models(self) -> dict:
        """Lazy-load models (expensive operation)."""
        with self._lock:
            self._cancel_unload_timer()
            if self._models is None:
                logger.info("Loading marker models into memory...")
                self._models = create_model_dict()
                logger.info("Marker models loaded successfully")
            self._last_used = time.time()
            return self._models

    def _cancel_unload_timer(self) -> None:
        """Cancel any pending unload timer."""
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def _schedule_unload(self) -> None:
        """Schedule model unload after idle timeout."""
        self._cancel_unload_timer()
        self._unload_timer = threading.Timer(MODEL_IDLE_TIMEOUT_SEC, self._unload_models)
        self._unload_timer.daemon = True
        self._unload_timer.start()
        logger.debug(f"Scheduled model unload in {MODEL_IDLE_TIMEOUT_SEC}s")

    def _unload_models(self) -> None:
        """Unload models from memory after idle timeout."""
        with self._lock:
            if self._models is None:
                return
            # Check if still idle
            if self._last_used and (time.time() - self._last_used) < MODEL_IDLE_TIMEOUT_SEC:
                # Used recently, reschedule
                self._schedule_unload()
                return
            logger.info("Unloading marker models after idle timeout...")
            self._models = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Marker models unloaded, memory freed")

    def cleanup(self) -> None:
        """Clean up intermediate memory after a job (keeps models loaded)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # Schedule unload timer
        self._schedule_unload()

    def convert(
        self,
        file_path: str,
        quality: str = "balanced",
        markdown_only: bool = False,
        langs: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Convert a document and return all output formats.

        Args:
            file_path: Path relative to input_dir
            quality: Quality preset (fast, balanced, quality)
            markdown_only: If True, return only markdown (smaller response)
            langs: List of languages for OCR (default: ["English"])

        Returns:
            Dict with markdown, json, chunks, and metadata
        """
        langs = langs or ["English"]
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])

        # Build full paths
        input_path = Path(self.settings.input_dir) / file_path
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Set environment for Marker
        self._configure_environment(preset, langs)

        # Run conversion
        start_time = time.time()
        result = self._run_conversion(input_path, preset, langs, markdown_only)
        processing_time = time.time() - start_time

        return {
            "markdown": result["markdown"],
            "json": result.get("json"),
            "chunks": result.get("chunks"),
            "metadata": {
                "input_file": file_path,
                "quality_preset": quality,
                "languages": langs,
                "pages": result.get("pages", 0),
                "ocr_method": result.get("ocr_method", "unknown"),
                "processing_time_sec": round(processing_time, 2),
                "force_ocr": preset["force_ocr"],
                "use_llm": preset["use_llm"],
            },
        }

    def _configure_environment(self, preset: dict, langs: list[str]) -> None:
        """Set environment variables for Marker."""
        settings = self.settings

        # Batch sizes
        os.environ["INFERENCE_RAM"] = str(settings.inference_ram)
        os.environ["RECOGNITION_BATCH_SIZE"] = str(settings.recognition_batch_size)
        os.environ["DETECTOR_BATCH_SIZE"] = str(settings.detector_batch_size)
        os.environ["LAYOUT_BATCH_SIZE"] = str(settings.layout_batch_size)
        os.environ["TABLE_REC_BATCH_SIZE"] = str(settings.table_rec_batch_size)

        # OCR settings
        if preset["force_ocr"]:
            os.environ["OCR_ALL_PAGES"] = "true"

        # Language
        os.environ["DEFAULT_LANG"] = langs[0] if langs else "English"

        # Google API key for LLM mode
        if preset["use_llm"] and settings.google_api_key:
            os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    def _run_conversion(
        self,
        input_path: Path,
        preset: dict,
        langs: list[str],
        markdown_only: bool = False,
    ) -> dict[str, Any]:
        """Run the actual Marker conversion and return all formats."""
        config = {
            "force_ocr": preset["force_ocr"],
            "use_llm": preset["use_llm"],
            "batch_multiplier": preset["batch_multiplier"],
            "languages": langs,
        }

        # Create converter without specifying renderer (we'll render manually)
        converter = PdfConverter(
            artifact_dict=self._get_models(),
            config=config,
        )

        # Build document once (expensive OCR/detection step)
        with converter.filepath_to_str(str(input_path)) as temp_path:
            document = converter.build_document(temp_path)
            pages = len(document.pages)

            # Create markdown renderer and render
            md_renderer = converter.resolve_dependencies(MarkdownRenderer)
            md_output = md_renderer(document)
            markdown = md_output.markdown

            # If markdown_only, skip chunk rendering
            if markdown_only:
                return {
                    "markdown": markdown,
                    "json": None,
                    "chunks": None,
                    "pages": pages,
                    "ocr_method": "surya" if preset["force_ocr"] else "native",
                }

            # Create chunk renderer and render (for RAG-ready chunks)
            chunk_renderer = converter.resolve_dependencies(ChunkRenderer)
            chunk_output = chunk_renderer(document)

            # Extract chunks as list of dicts (JSON-serializable)
            chunks = [
                block.model_dump(mode="json", exclude={"images"})
                for block in chunk_output.blocks
            ]

            # Build JSON structure from chunk output
            json_data = {
                "blocks": chunks,
                "page_info": chunk_output.page_info,
                "metadata": chunk_output.metadata,
            }

        return {
            "markdown": markdown,
            "json": json_data,
            "chunks": chunks,
            "pages": pages,
            "ocr_method": "surya" if preset["force_ocr"] else "native",
        }


# Singleton processor instance for Celery workers
_processor: MarkerProcessor | None = None


def get_processor() -> MarkerProcessor:
    """Get or create processor instance."""
    global _processor
    if _processor is None:
        _processor = MarkerProcessor()
    return _processor
