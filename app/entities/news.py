from pydantic import BaseModel, HttpUrl
from datetime import datetime


class News(BaseModel):
    id: int
    title: str
    issued_at: datetime
    modified_at: datetime
    url: HttpUrl
