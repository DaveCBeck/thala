"""Response parsing utilities for Marker API."""

from typing import Any

from workflows.shared.marker_client.types import MarkerJobResult


def parse_job_result(data: dict[str, Any]) -> MarkerJobResult:
    """Parse job result data into MarkerJobResult."""
    return MarkerJobResult(**data)
