"""Types and constants for paper acquisition."""

# Processing concurrency can be higher than pipeline concurrency
# since marker has its own queue and multiple workers
MAX_PROCESSING_CONCURRENT = 4

# Queue size for streaming pipeline - balances memory vs latency
# ~8 PDFs × 50MB = ~400MB buffer max
PROCESSING_QUEUE_SIZE = 8

# Timeout for OA URL downloads
OA_DOWNLOAD_TIMEOUT = 60.0
