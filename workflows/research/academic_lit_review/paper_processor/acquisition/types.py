"""Types and constants for paper acquisition."""

# Two-stage pipeline constants
# Stage 1: Marker (PDF→markdown) - GPU-bound, fast
MAX_MARKER_CONCURRENT = 4
MARKER_QUEUE_SIZE = 8  # ~8 PDFs × 50MB = ~400MB buffer max

# Stage 2: LLM workflow (summaries, metadata, chapters) - IO-bound with batch API delays
MAX_LLM_CONCURRENT = 4
# LLM queue is unbounded since markdown text is ~100KB-1MB vs PDFs at ~50MB

# Legacy constants (for backward compatibility with old acquisition module)
MAX_PROCESSING_CONCURRENT = MAX_MARKER_CONCURRENT
PROCESSING_QUEUE_SIZE = MARKER_QUEUE_SIZE

# Timeout for OA URL downloads
OA_DOWNLOAD_TIMEOUT = 60.0
