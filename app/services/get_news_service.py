import pandas as pd

from app.utils.settings import settings
from app.entities.news import News

class GetNewsService:
    def __init__(self):
        self.df_news = pd.read_csv(settings.RESOURCES_PATH + 'news.csv', encoding='utf-8')

    def execute(self, news_id: int) -> News:
        news_dict = self.df_news.loc[news_id].to_dict()
        news_out_dto = News(
            id=news_id,
            title=news_dict['title'],
            issued_at=news_dict['issued'],
            modified_at=news_dict['modified'],
            url=news_dict['url']
        )
        return news_out_dto
