
from config.settings import BaseConfig


class ProductionConfig(BaseConfig):
    """Production settings are supplied through environment variables."""

    DEBUG = False
