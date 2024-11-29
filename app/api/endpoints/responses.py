from fastapi import APIRouter, HTTPException
from app.schemas.user_message import UserMessage
from app.services.openai_service import OpenAIService
from app.services.gemini_service import GeminiService

# Initialize API Router
router = APIRouter()

# In-memory session storage
sessions = {}


@router.post("/generate-response", summary="Generate AI response")
async def generate_response(user_message: UserMessage):
    """
    Generate a response using OpenAI or Gemini models.

    Args:
        user_message (UserMessage): Input message with session ID and model type.

    Returns:
        dict: Assistant's response.
    """
    try:
        # Initialize session if it doesn't exist
        if user_message.session_id not in sessions:
            sessions[user_message.session_id] = [
                {"role": "user", "content": "You are a helpful assistant."}
            ]

        # Add user message to session
        sessions[user_message.session_id].append(
            {"role": "user", "content": user_message.message}
        )

        # Select service based on model prefix
        if user_message.model.startswith("gemini"):
            service = GeminiService()
            assistant_response = service.get_response(
                user_message, sessions[user_message.session_id]
            )
        elif user_message.model.startswith("gpt-"):
            service = OpenAIService()
            assistant_response = service.get_response(
                user_message, sessions[user_message.session_id]
            )
        else:
            raise ValueError(
                f"Invalid model specified: {user_message.model}. Supported models: gemini, gpt-*."
            )

        # Append assistant's response to session
        sessions[user_message.session_id].append(
            {"role": "assistant", "content": assistant_response}
        )

        # Return response
        return {"response": assistant_response}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
