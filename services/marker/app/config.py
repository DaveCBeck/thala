"""Configuration for Marker document processing service."""

from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Celery configuration
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"

    # Google API key for LLM mode (Gemini 2.0 Flash)
    google_api_key: str = ""

    # File paths
    input_dir: str = "/data/input"
    output_dir: str = "/data/output"

    # Marker batch settings (tuned for RTX 4060 Ti 16GB)
    inference_ram: int = 16
    recognition_batch_size: int = 256
    detector_batch_size: int = 24
    layout_batch_size: int = 32
    table_rec_batch_size: int = 48

    class Config:
        env_file = ".env"
        extra = "ignore"


# Quality presets for different document types
QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "batch_multiplier": 4,
        "force_ocr": False,
        "use_llm": False,
        "description": "Fast processing for good quality digital documents",
    },
    "balanced": {
        "batch_multiplier": 2,
        "force_ocr": False,
        "use_llm": False,
        "description": "Balanced quality/speed for general documents",
    },
    "quality": {
        "batch_multiplier": 1,
        "force_ocr": True,
        "use_llm": True,
        "description": "Highest quality for scanned/degraded documents with tables",
    },
}

# Supported output formats
OUTPUT_FORMATS = ["markdown", "json", "html", "chunks"]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
