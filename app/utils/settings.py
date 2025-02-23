import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    RESOURCES_PATH = os.getenv('RESOURCES_PATH')


settings = Settings()
