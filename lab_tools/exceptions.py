class NotFittedError(Exception):

    def __init__(
        self,
        message: str | None = None,
    ):
        if message is None:
            message = "This model instance is not fitted yet. Call `.fit()` first"
        super().__init__(message)

