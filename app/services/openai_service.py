from app.core.config import settings
import requests
from fastapi import HTTPException

class OpenAIService:
    def get_response(self, user_message, session):
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": user_message.model,
            "messages": session,
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
