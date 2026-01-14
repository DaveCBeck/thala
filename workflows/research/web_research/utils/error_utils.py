"""Error handling utilities."""


def create_error_result(
    node_name: str, error: Exception, status: str = "failed", **kwargs
) -> dict:
    """Create standardized error result dict."""
    return {
        "current_status": status,
        "errors": [{"node": node_name, "error": str(error)}],
        **kwargs,
    }
