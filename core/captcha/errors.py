"""Captcha-specific exceptions."""


class CaptchaSolveError(Exception):
    """Captcha solve failed permanently."""

    def __init__(self, error_code: str, error_description: str):
        self.error_code = error_code
        self.error_description = error_description
        super().__init__(f"{error_code}: {error_description}")
