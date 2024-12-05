from app.core.config import settings
from openai import OpenAI
from fastapi import HTTPException


class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def get_response(self, user_message, session):
        """
        Generate a response using the Chat Completions API via the OpenAI client.
        """
        try:
            response = self.client.chat.completions.create(
                model=user_message.model,
                messages=session,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

    def get_stream_response(self, user_message, session):
        """
        Generate a streaming response using the Chat Completions API.
        """
        try:
            openai_stream = self.client.chat.completions.create(
                model=user_message.model,
                messages=session,
                stream=True,  # Enable streaming
            )
            for event in openai_stream:
                # Access the "delta" attribute safely
                delta = event.choices[0].delta
                if hasattr(delta, "content"):
                    yield delta.content  # Stream the content
        except Exception as e:
            yield f"Error: {str(e)}"
