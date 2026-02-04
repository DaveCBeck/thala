"""Types and constants for paper acquisition."""

# Two-stage pipeline constants
# Stage 1: Marker (PDF→markdown) - Marker service has its own Redis queue (Celery)
# that handles job queuing internally. We use unbounded asyncio queues here since
# they just hold file paths or markdown text (not PDF bytes in memory).

# Stage 2: LLM workflow (summaries, metadata, chapters) - IO-bound with batch API delays
MAX_LLM_CONCURRENT = 4

# Timeout for OA URL downloads
OA_DOWNLOAD_TIMEOUT = 60.0
