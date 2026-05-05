class ValidationError(Exception):
    def __init__(self, err_msg: str) -> None:
        self.err_msg = err_msg
        super().__init__(self.err_msg)

    def __str__(self) -> str:
        return f"{self.err_msg}"
