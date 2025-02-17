from fastapi import APIRouter
from starlette import status

router = APIRouter(prefix='/news')


@router.get('',
            description='Endpoint de testes',
            status_code=status.HTTP_200_OK)
def get_news() -> bool:
    return True
