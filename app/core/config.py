import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

    # Ensure both API keys are set
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please check your environment variables."
        )
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Please check your environment variables."
        )


# Create a singleton settings instance
settings = Settings()
