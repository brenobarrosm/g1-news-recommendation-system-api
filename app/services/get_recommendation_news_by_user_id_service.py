import torch
import torch.nn.functional as F

from app.services.get_news_service import GetNewsService
from app.entities.news import News
from app.utils.settings import settings
from app.utils.model import get_model
from app.exceptions.user_not_found_exception import UserNotFoundException
from app.exceptions.top_k_exceeded_limit_exception import TopKExceededLimitException


class GetRecommendationNewsByUserIdService:

    def __init__(self):
        self.get_news_service = GetNewsService()
        self.model = get_model(num_users=577942,
                               num_items=255603,
                               embedding_dim=64)
        self.limit = 20

    def execute(self, user_id: int, top_k: int) -> list[News]:
        self.raise_if_user_not_found(user_id)
        self.raise_if_top_k_exceeded_limit(top_k)
        items_recommended_ids: list[int] = self.recommend_items(user_id, top_k)
        items_recommended: list[News] = []
        for item_id in items_recommended_ids:
            items_recommended.append(self.get_news_service.execute(item_id))
        return items_recommended

    def recommend_items(self, user_id: int, top_k: int) -> list:
        user_embeddings = self.load_saved_user_embeddings()
        item_embeddings = self.load_saved_item_embeddings()
        user_vector = user_embeddings[user_id].unsqueeze(0)
        scores = F.cosine_similarity(user_vector, item_embeddings)
        top_items = torch.argsort(scores, descending=True)[:top_k]
        return top_items.tolist()

    @staticmethod
    def load_saved_item_embeddings():
        item_embeddings = torch.load(settings.RESOURCES_PATH + "item_embeddings.pt",
                                     map_location=torch.device('cpu'))
        return item_embeddings

    @staticmethod
    def load_saved_user_embeddings():
        user_embeddings = torch.load(settings.RESOURCES_PATH + "user_embeddings.pt",
                                     map_location=torch.device('cpu'))
        return user_embeddings

    def raise_if_user_not_found(self, user_id: int):
        total_user_embeddings: int = len(self.load_saved_user_embeddings())
        if (user_id < 0 and user_id > total_user_embeddings):
            raise UserNotFoundException

    def raise_if_top_k_exceeded_limit(self, top_k: int):
        if top_k > self.limit:
            raise TopKExceededLimitException
