from fastapi import APIRouter, Query
from starlette import status

from app.services.get_news_service import GetNewsService
from app.services.get_recommendation_news_by_user_id_service import GetRecommendationNewsByUserIdService

router = APIRouter(prefix='/news')

get_news_service = GetNewsService()
get_recommendation_news_by_user_id_service = GetRecommendationNewsByUserIdService()


@router.get('/{news_id}',
            description='Busca uma notícia pelo seu ID.',
            status_code=status.HTTP_200_OK)
def get_news(news_id: int):
    return get_news_service.execute(news_id)

@router.get('/recommendation/{user_id}',
            description='Retorna uma lista de notícias recomendados para um usuário específico.',
            status_code=status.HTTP_200_OK)
def get_recommended_news(user_id: int,
             top_k: int = Query(5, description="Número de notícias a serem retornadas (máximo de 20)")):
    return get_recommendation_news_by_user_id_service.execute(user_id, top_k)
