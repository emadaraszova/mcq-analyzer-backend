from app.core.config import settings
import google.generativeai as genai
from fastapi import HTTPException
from starlette.responses import StreamingResponse


class GeminiService:
    def __init__(self):
        
        api_key = settings.GEMINI_API_KEY
        genai.configure(api_key=api_key)

    async def get_response(self, user_message, session, stream: bool):
        gemini_history = [{"role": entry["role"], "parts": [entry["content"]]} for entry in session]

        try:
            gemini_model = genai.GenerativeModel(model_name=user_message.model)
            chat_session = gemini_model.start_chat(history=gemini_history)

            if stream:
                async def stream_response():
                    response = chat_session.send_message(user_message.message, stream=True)
                    for chunk in response:
                        yield chunk.text
                return StreamingResponse(stream_response(), media_type="text/plain")
            else:
                response = chat_session.send_message(user_message.message)
                return response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini Error: {str(e)}")