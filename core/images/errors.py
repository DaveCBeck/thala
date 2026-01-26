"""Exception classes for image aggregator."""


class ImageError(Exception):
    """Base image service exception."""

    def __init__(self, message: str, provider: str | None = None):
        self.message = message
        self.provider = provider
        super().__init__(message)


class NoResultsError(ImageError):
    """No images found for query."""

    pass


class RateLimitError(ImageError):
    """API rate limit exceeded."""

    pass


class ProviderError(ImageError):
    """Provider-specific error."""

    pass
