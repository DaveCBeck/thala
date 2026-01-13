"""Prompts for deep research workflow."""

from workflows.research.web_research.prompts.utils import get_today_str
from workflows.research.web_research.prompts.clarification import (
    CLARIFY_INTENT_SYSTEM,
    CLARIFY_INTENT_HUMAN,
)
from workflows.research.web_research.prompts.brief import (
    CREATE_BRIEF_SYSTEM,
    CREATE_BRIEF_HUMAN,
)
from workflows.research.web_research.prompts.planning import (
    ITERATE_PLAN_SYSTEM,
    ITERATE_PLAN_HUMAN,
)
from workflows.research.web_research.prompts.supervision import (
    SUPERVISOR_SYSTEM_CACHED,
    SUPERVISOR_USER_TEMPLATE,
    SUPERVISOR_DIFFUSION_SYSTEM,
)
from workflows.research.web_research.prompts.researcher import RESEARCHER_SYSTEM
from workflows.research.web_research.prompts.compression import (
    COMPRESS_RESEARCH_SYSTEM_CACHED,
    COMPRESS_RESEARCH_USER_TEMPLATE,
    COMPRESS_RESEARCH_SYSTEM,
    COMPRESS_WEB_RESEARCH_SYSTEM,
    COMPRESS_ACADEMIC_RESEARCH_SYSTEM,
    COMPRESS_BOOK_RESEARCH_SYSTEM,
)
from workflows.research.web_research.prompts.queries import (
    GENERATE_WEB_QUERIES_SYSTEM,
    GENERATE_ACADEMIC_QUERIES_SYSTEM,
    GENERATE_BOOK_QUERIES_SYSTEM,
)
from workflows.research.web_research.prompts.reporting import (
    FINAL_REPORT_SYSTEM_STATIC,
    FINAL_REPORT_USER_TEMPLATE,
    FINAL_REPORT_SYSTEM,
    FINAL_REPORT_HUMAN,
    REFINE_DRAFT_SYSTEM,
)
from workflows.research.web_research.prompts.translator import (
    translate_prompt,
    get_translated_prompt,
    clear_translation_cache,
    PROMPT_TRANSLATION_SYSTEM,
)

__all__ = [
    # Utils
    "get_today_str",
    # Clarification
    "CLARIFY_INTENT_SYSTEM",
    "CLARIFY_INTENT_HUMAN",
    # Brief
    "CREATE_BRIEF_SYSTEM",
    "CREATE_BRIEF_HUMAN",
    # Planning
    "ITERATE_PLAN_SYSTEM",
    "ITERATE_PLAN_HUMAN",
    # Supervision
    "SUPERVISOR_SYSTEM_CACHED",
    "SUPERVISOR_USER_TEMPLATE",
    "SUPERVISOR_DIFFUSION_SYSTEM",
    # Researcher
    "RESEARCHER_SYSTEM",
    # Compression
    "COMPRESS_RESEARCH_SYSTEM_CACHED",
    "COMPRESS_RESEARCH_USER_TEMPLATE",
    "COMPRESS_RESEARCH_SYSTEM",
    "COMPRESS_WEB_RESEARCH_SYSTEM",
    "COMPRESS_ACADEMIC_RESEARCH_SYSTEM",
    "COMPRESS_BOOK_RESEARCH_SYSTEM",
    # Queries
    "GENERATE_WEB_QUERIES_SYSTEM",
    "GENERATE_ACADEMIC_QUERIES_SYSTEM",
    "GENERATE_BOOK_QUERIES_SYSTEM",
    # Reporting
    "FINAL_REPORT_SYSTEM_STATIC",
    "FINAL_REPORT_USER_TEMPLATE",
    "FINAL_REPORT_SYSTEM",
    "FINAL_REPORT_HUMAN",
    "REFINE_DRAFT_SYSTEM",
    # Translator
    "translate_prompt",
    "get_translated_prompt",
    "clear_translation_cache",
    "PROMPT_TRANSLATION_SYSTEM",
]
