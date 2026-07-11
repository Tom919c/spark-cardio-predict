from config.settings import BaseConfig


class DevelopmentConfig(BaseConfig):
    """Development defaults keep Flask local and debug friendly."""

    DEBUG = True
