"""Prompts for deep research workflow."""

# Re-export all original prompts
from workflows.research.prompts.base import (
    get_today_str,
    CLARIFY_INTENT_SYSTEM,
    CLARIFY_INTENT_HUMAN,
    CREATE_BRIEF_SYSTEM,
    CREATE_BRIEF_HUMAN,
    ITERATE_PLAN_SYSTEM,
    ITERATE_PLAN_HUMAN,
    SUPERVISOR_SYSTEM_CACHED,
    SUPERVISOR_USER_TEMPLATE,
    SUPERVISOR_DIFFUSION_SYSTEM,
    RESEARCHER_SYSTEM,
    COMPRESS_RESEARCH_SYSTEM_CACHED,
    COMPRESS_RESEARCH_USER_TEMPLATE,
    COMPRESS_RESEARCH_SYSTEM,
    # Specialized compression prompts
    COMPRESS_WEB_RESEARCH_SYSTEM,
    COMPRESS_ACADEMIC_RESEARCH_SYSTEM,
    COMPRESS_BOOK_RESEARCH_SYSTEM,
    FINAL_REPORT_SYSTEM_STATIC,
    FINAL_REPORT_USER_TEMPLATE,
    FINAL_REPORT_SYSTEM,
    FINAL_REPORT_HUMAN,
    REFINE_DRAFT_SYSTEM,
)

# Export translation utilities
from workflows.research.prompts.translator import (
    translate_prompt,
    get_translated_prompt,
    clear_translation_cache,
    PROMPT_TRANSLATION_SYSTEM,
)

__all__ = [
    # Base prompts
    "get_today_str",
    "CLARIFY_INTENT_SYSTEM",
    "CLARIFY_INTENT_HUMAN",
    "CREATE_BRIEF_SYSTEM",
    "CREATE_BRIEF_HUMAN",
    "ITERATE_PLAN_SYSTEM",
    "ITERATE_PLAN_HUMAN",
    "SUPERVISOR_SYSTEM_CACHED",
    "SUPERVISOR_USER_TEMPLATE",
    "SUPERVISOR_DIFFUSION_SYSTEM",
    "RESEARCHER_SYSTEM",
    "COMPRESS_RESEARCH_SYSTEM_CACHED",
    "COMPRESS_RESEARCH_USER_TEMPLATE",
    "COMPRESS_RESEARCH_SYSTEM",
    # Specialized compression prompts
    "COMPRESS_WEB_RESEARCH_SYSTEM",
    "COMPRESS_ACADEMIC_RESEARCH_SYSTEM",
    "COMPRESS_BOOK_RESEARCH_SYSTEM",
    "FINAL_REPORT_SYSTEM_STATIC",
    "FINAL_REPORT_USER_TEMPLATE",
    "FINAL_REPORT_SYSTEM",
    "FINAL_REPORT_HUMAN",
    "REFINE_DRAFT_SYSTEM",
    # Translator utilities
    "translate_prompt",
    "get_translated_prompt",
    "clear_translation_cache",
    "PROMPT_TRANSLATION_SYSTEM",
]
