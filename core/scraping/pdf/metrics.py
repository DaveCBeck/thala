"""Prometheus metrics for PDF processing and routing.

Provides observability into routing effectiveness and processing performance.
"""

from prometheus_client import Counter, Histogram

# Route decisions
ROUTE_COUNTER = Counter(
    "pdf_route_total",
    "Documents by processing route",
    ["route"],  # cpu, gpu, gpu_fallback
)

# Complexity tiers
COMPLEXITY_COUNTER = Counter(
    "pdf_complexity_total",
    "Documents by complexity tier",
    ["tier"],  # light, mixed, heavy
)

# CPU fallback rate
FALLBACK_COUNTER = Counter(
    "pdf_cpu_fallback_total",
    "CPU extractions that fell back to GPU",
)

# Processing duration by route
PROCESSING_HISTOGRAM = Histogram(
    "pdf_processing_seconds",
    "Processing time by route",
    ["route"],
    buckets=[1, 5, 15, 30, 60, 120, 300, 600],
)

# Analysis duration
ANALYSIS_HISTOGRAM = Histogram(
    "pdf_analysis_seconds",
    "Document analysis time",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
)

# Page counts by route
PAGE_COUNT_HISTOGRAM = Histogram(
    "pdf_page_count",
    "Page counts by route",
    ["route"],
    buckets=[1, 5, 10, 25, 50, 100, 200, 500],
)


def record_route_decision(route: str, complexity: str, page_count: int) -> None:
    """Record routing decision metrics.

    Args:
        route: Processing route (cpu, gpu, gpu_fallback)
        complexity: Complexity tier (light, mixed, heavy)
        page_count: Number of pages in document
    """
    ROUTE_COUNTER.labels(route=route).inc()
    COMPLEXITY_COUNTER.labels(tier=complexity).inc()
    PAGE_COUNT_HISTOGRAM.labels(route=route).observe(page_count)


def record_cpu_fallback() -> None:
    """Record a CPU-to-GPU fallback event."""
    FALLBACK_COUNTER.inc()


def record_processing_time(route: str, duration_seconds: float) -> None:
    """Record processing duration.

    Args:
        route: Processing route (cpu, gpu, gpu_fallback)
        duration_seconds: Processing time in seconds
    """
    PROCESSING_HISTOGRAM.labels(route=route).observe(duration_seconds)


def record_analysis_time(duration_seconds: float) -> None:
    """Record document analysis duration.

    Args:
        duration_seconds: Analysis time in seconds
    """
    ANALYSIS_HISTOGRAM.observe(duration_seconds)
