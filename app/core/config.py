"""Application settings and environment variable configuration."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application configuration loaded from environment variables.

    Attributes:
        OPENAI_API_KEY: API key for OpenAI models.
        GEMINI_API_KEY: API key for Google Gemini models.
        E_INFRA_API_KEY: API key for E-INFRA service.
        E_INFRA_BASE_URL: Base URL for the E-INFRA chat API.
        REDIS_URL: Redis connection string for task queues.
    """

    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    E_INFRA_API_KEY: str = Field(..., env="E_INFRA_API_KEY")
    E_INFRA_BASE_URL: str = Field(
        "https://chat.ai.e-infra.cz/api", env="E_INFRA_BASE_URL"
    )
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")

    class Config:
        """Pydantic settings configuration."""

        # Use .env from project root if available
        env_file = str(Path(__file__).resolve().parents[2] / ".env")
        env_file_encoding = "utf-8"


# Instantiate the settings object at import time
settings = Settings()
