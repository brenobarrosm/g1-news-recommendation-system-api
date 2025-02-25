import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    RESOURCES_PATH = 'resources/' # Pasta resources contém os modelos, porém são muito pesados para fazer upload no github


settings = Settings()
