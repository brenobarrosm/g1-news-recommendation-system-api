from fastapi import HTTPException
from pydantic import BaseModel

ERROR_MSG = 'TOP_K_EXCEEDED_LIMIT_EXCEPTION'


class TopKExceededLimitException(HTTPException):
    def __init__(self) -> None:
        self.status_code = 409
        self.detail = ERROR_MSG


class TopKExcedeedLimitModel(BaseModel):
    error_msg: str | None = ERROR_MSG
