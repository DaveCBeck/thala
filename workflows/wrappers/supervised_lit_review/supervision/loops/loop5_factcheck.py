"""Loop 5: Fact and reference checking with tool access.

DEPRECATED: This module has been refactored into the loop5/ package.
Import from workflows.wrappers.supervised_lit_review.supervision.loops.loop5 instead.
This file exists for backward compatibility only.
"""

from .loop5 import (
    Loop5State,
    create_loop5_graph,
    run_loop5_standalone,
    fact_check_node,
    reference_check_node,
    split_sections_node,
    validate_edits_node,
    apply_edits_node,
    flag_issues_node,
    finalize_node,
    filter_ambiguous_claims,
    select_model_tier_for_context,
    estimate_loop5_request_tokens,
    calculate_dynamic_char_budget,
    format_paper_summaries,
    format_citation_keys,
    HAIKU_MAX_TOKENS,
    SONNET_1M_MAX_TOKENS,
    SONNET_1M_THRESHOLD,
    RESPONSE_BUFFER_TOKENS,
)

__all__ = [
    "Loop5State",
    "create_loop5_graph",
    "run_loop5_standalone",
    "fact_check_node",
    "reference_check_node",
    "split_sections_node",
    "validate_edits_node",
    "apply_edits_node",
    "flag_issues_node",
    "finalize_node",
    "filter_ambiguous_claims",
    "select_model_tier_for_context",
    "estimate_loop5_request_tokens",
    "calculate_dynamic_char_budget",
    "format_paper_summaries",
    "format_citation_keys",
    "HAIKU_MAX_TOKENS",
    "SONNET_1M_MAX_TOKENS",
    "SONNET_1M_THRESHOLD",
    "RESPONSE_BUFFER_TOKENS",
]
