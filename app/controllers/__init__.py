from fastapi import FastAPI, Depends

from . import news_controller


class AppRouters:
    def __init__(self):
        self.app = None
        self.api_prefix = '/api'

    def start_router(self, app: FastAPI):
        self.app = app
        self.__include_routes()

    def __include_routes(self):
        self.app.include_router(router=news_controller.router, prefix=self.api_prefix, tags=['News'])


app_routers = AppRouters()
