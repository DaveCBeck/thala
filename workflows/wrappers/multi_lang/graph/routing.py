"""Routing functions for multi-language workflow conditional edges."""

from workflows.wrappers.multi_lang.state import MultiLangState


def route_after_language_selection(state: MultiLangState) -> str:
    """Route after language selection based on mode."""
    mode = state["input"]["mode"]
    if mode == "all_languages":
        return "check_relevance_batch"
    return "execute_next_language"


def route_language_loop(state: MultiLangState) -> str:
    """Route after checking if all languages are complete."""
    idx = state.get("current_language_index", 0)

    # Determine which list to use
    languages_to_process = state.get("languages_with_content") or state.get("target_languages", [])

    if idx < len(languages_to_process):
        return "execute_next_language"
    return "sonnet_analysis"
