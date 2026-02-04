"""Types and constants for paper acquisition."""

# Two-stage pipeline constants
# Stage 1: Marker (PDF→markdown) - Marker service has its own Redis queue (Celery)
# that handles job queuing internally. We use unbounded asyncio queues here since
# they just hold file paths or markdown text (not PDF bytes in memory).

# Stage 2: LLM workflow (summaries, metadata, chapters) - routed through central
# LLM broker which handles batching and concurrency internally.

# Timeout for OA URL downloads
OA_DOWNLOAD_TIMEOUT = 60.0
