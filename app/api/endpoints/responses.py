from fastapi import APIRouter, HTTPException
from app.schemas.user_message import UserMessage
from app.services.openai_service import OpenAIService
from app.services.gemini_service import GeminiService

# Initialize API Router
router = APIRouter()

# In-memory session storage
sessions = {}


@router.post("/generate-response", summary="Generate AI response")
async def generate_response(user_message: UserMessage, stream: bool = True):
    try:
        if user_message.session_id not in sessions:
            sessions[user_message.session_id] = [{"role": "user", "content": "You are a helpful assistant."}]
        sessions[user_message.session_id].append({"role": "user", "content": user_message.message})

        if user_message.model.startswith("gemini"):
            service = GeminiService()
        elif user_message.model.startswith("gpt-"):
            service = OpenAIService()
        else:
            raise ValueError(f"Invalid model specified: {user_message.model}.")

        response = await service.get_response(user_message, sessions[user_message.session_id], stream=stream)

        if not stream:
            sessions[user_message.session_id].append({"role": "assistant", "content": response})

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
