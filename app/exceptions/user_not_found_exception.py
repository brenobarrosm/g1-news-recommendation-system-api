from fastapi import HTTPException
from pydantic import BaseModel

ERROR_MSG = 'USER_NOT_FOUND_EXCEPTION'


class UserNotFoundException(HTTPException):
    def __init__(self) -> None:
        self.status_code = 404
        self.detail = ERROR_MSG


class UserNotFoundModel(BaseModel):
    error_msg: str | None = ERROR_MSG
