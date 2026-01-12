"""State reducer functions for academic literature review workflow."""


def merge_dicts(existing: dict, new: dict) -> dict:
    """Merge two dicts, with new values overwriting existing."""
    return {**existing, **new}


def merge_paper_summaries(existing: dict, new: dict) -> dict:
    """Merge paper summary dicts, preferring existing summaries if present."""
    merged = existing.copy()
    for doi, summary in new.items():
        if doi not in merged:
            merged[doi] = summary
    return merged
