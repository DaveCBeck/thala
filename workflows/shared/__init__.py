"""Shared utilities for document processing workflows."""

from .marker_client import MarkerClient, MarkerJobResult
from .text_utils import (
    chunk_by_headings,
    count_words,
    estimate_pages,
    get_first_n_pages,
    get_last_n_pages,
)
from .llm_utils import (
    ModelTier,
    analyze_with_thinking,
    extract_json,
    get_llm,
    summarize_text,
)
from .batch_processor import (
    BatchProcessor,
    BatchRequest,
    BatchResult,
    get_batch_processor,
)
from .persistent_cache import (
    get_cached,
    set_cached,
    clear_cache,
    get_cache_stats,
    compute_file_hash,
)
from .async_utils import (
    run_with_concurrency,
    gather_with_error_collection,
)
from .retry_utils import with_retry
from .node_utils import safe_node_execution, StateUpdater
from .llm_utils.response_parsing import (
    extract_json_from_response,
    extract_response_content,
)
from .token_utils import (
    estimate_tokens_fast,
    count_tokens_accurate,
    estimate_tokens,
    estimate_request_tokens,
    check_token_budget,
    select_model_for_context,
    get_safe_limit_for_model,
    TokenBudgetExceeded,
    HAIKU_SAFE_LIMIT,
    SONNET_SAFE_LIMIT,
    SONNET_1M_SAFE_LIMIT,
    CHARS_PER_TOKEN,
    DEFAULT_RESPONSE_BUFFER,
)

__all__ = [
    # Marker client
    "MarkerClient",
    "MarkerJobResult",
    # Text utilities
    "chunk_by_headings",
    "count_words",
    "estimate_pages",
    "get_first_n_pages",
    "get_last_n_pages",
    # LLM utilities
    "ModelTier",
    "analyze_with_thinking",
    "extract_json",
    "get_llm",
    "summarize_text",
    # Batch processing
    "BatchProcessor",
    "BatchRequest",
    "BatchResult",
    "get_batch_processor",
    # Persistent caching
    "get_cached",
    "set_cached",
    "clear_cache",
    "get_cache_stats",
    "compute_file_hash",
    # Async utilities
    "run_with_concurrency",
    "gather_with_error_collection",
    # Retry utilities
    "with_retry",
    # Node utilities
    "safe_node_execution",
    "StateUpdater",
    # Response parsing
    "extract_json_from_response",
    "extract_response_content",
    # Token utilities
    "estimate_tokens_fast",
    "count_tokens_accurate",
    "estimate_tokens",
    "estimate_request_tokens",
    "check_token_budget",
    "select_model_for_context",
    "get_safe_limit_for_model",
    "TokenBudgetExceeded",
    "HAIKU_SAFE_LIMIT",
    "SONNET_SAFE_LIMIT",
    "SONNET_1M_SAFE_LIMIT",
    "CHARS_PER_TOKEN",
    "DEFAULT_RESPONSE_BUFFER",
]
