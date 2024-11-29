from app.core.config import settings
import google.generativeai as genai
from fastapi import HTTPException

class GeminiService:
    def __init__(self):
        # Use the GEMINI_API_KEY from settings
        api_key = settings.GEMINI_API_KEY
        genai.configure(api_key=api_key)

    def get_response(self, user_message, session):
        gemini_history = [
            {
                "role": entry["role"],
                "parts": [entry["content"]]
            }
            for entry in session
            if entry["role"] in ["user", "assistant"]
        ]

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        try:
            gemini_model = genai.GenerativeModel(
                model_name=user_message.model,
                generation_config=generation_config
            )
            chat_session = gemini_model.start_chat(history=gemini_history)
            response = chat_session.send_message(user_message.message)
            return response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")
