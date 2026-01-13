"""Language selection node for multi-lingual research workflow."""

import logging

from workflows.wrappers.multi_lang.state import MultiLangState
from workflows.shared.language import get_language_config, is_supported_language

logger = logging.getLogger(__name__)


MAJOR_10_LANGUAGES = [
    "en",  # English - baseline
    "zh",  # Mandarin Chinese - largest non-English internet
    "es",  # Spanish - 500M+ speakers
    "de",  # German - strong academic tradition
    "fr",  # French - Francophone world
    "ja",  # Japanese - major research output
    "pt",  # Portuguese - Brazil's research growth
    "ru",  # Russian - distinct traditions
    "ar",  # Arabic - Middle East perspectives
    "ko",  # Korean - tech/science leader
]


async def select_languages(state: MultiLangState) -> dict:
    """
    Select target languages based on mode.

    Mode 1 (set_languages): Use user-specified languages
    Mode 2 (all_languages): Start with MAJOR_10_LANGUAGES

    Also builds language_configs dict using get_language_config from shared/language.
    """
    mode = state["input"]["mode"]

    # Select languages based on mode
    if mode == "set_languages":
        requested_languages = state["input"].get("languages", [])
        if not requested_languages:
            logger.error("No languages provided for set_languages mode")
            return {
                "current_phase": "language_selection",
                "current_status": "Error: No languages provided for set_languages mode",
                "errors": [
                    {
                        "phase": "language_selection",
                        "error": "set_languages mode requires languages to be specified",
                    }
                ],
            }

        # Validate all requested languages
        valid_languages = []
        invalid_languages = []
        for code in requested_languages:
            if is_supported_language(code):
                valid_languages.append(code)
            else:
                invalid_languages.append(code)

        if invalid_languages:
            logger.warning(f"Invalid language codes filtered out: {', '.join(invalid_languages)}")

        if not valid_languages:
            logger.error(f"All provided language codes are invalid: {requested_languages}")
            return {
                "current_phase": "language_selection",
                "current_status": "Error: No valid language codes provided",
                "errors": [
                    {
                        "phase": "language_selection",
                        "error": f"All provided language codes are invalid: {requested_languages}",
                    }
                ],
            }

        selected_languages = valid_languages
    else:  # all_languages mode
        selected_languages = MAJOR_10_LANGUAGES.copy()

    # Ensure "en" is first if present
    if "en" in selected_languages:
        selected_languages.remove("en")
        selected_languages.insert(0, "en")

    # Build language configs
    language_configs = {}
    for code in selected_languages:
        config = get_language_config(code)
        if config:
            language_configs[code] = config

    language_names = ", ".join(
        config["name"] for config in language_configs.values()
    )

    logger.info(f"Selected {len(selected_languages)} languages: {language_names}")

    return {
        "target_languages": selected_languages,
        "language_configs": language_configs,
        "current_language_index": 0,
        "languages_completed": [],
        "languages_failed": [],
        "current_phase": "language_selection",
        "current_status": f"Languages selected: {language_names}",
    }
