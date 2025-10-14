from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")

    class Config:
        env_file = str(Path(__file__).resolve().parents[2] / ".env")
        env_file_encoding = "utf-8"
    
settings = Settings()
