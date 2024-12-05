from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.user_message import UserMessage
from app.services.openai_service import OpenAIService
from app.services.gemini_service import GeminiService

# Initialize API Router
router = APIRouter()

# In-memory session storage
sessions = {}


@router.post("/generate-response", summary="Generate AI response")
async def generate_response(user_message: UserMessage, stream: bool = True):
    """
    Generate a response using OpenAI or Gemini models, with optional streaming.

    Args:
        user_message (UserMessage): Input message with session ID and model type.
        stream (bool): Whether to stream the response or return it as a single chunk.

    Returns:
        StreamingResponse or dict: Assistant's response.
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
            service = GeminiService()  # No project_id needed
        elif user_message.model.startswith("gpt-"):
            service = OpenAIService()
        else:
            raise ValueError(
                f"Invalid model specified: {user_message.model}. Supported models: gemini, gpt-*."
            )

        if stream:
            # Stream responses
            def response_generator():
                try:
                    for chunk in service.get_stream_response(
                        user_message, sessions[user_message.session_id]
                    ):
                        if chunk:
                            print(f"Yielding chunk: {chunk}")
                            yield f"data: {chunk}\n\n"
                except Exception as e:
                    print(f"Streaming error: {e}")
                    yield f"data: [Error] {str(e)}\n\n"

            return StreamingResponse(
                response_generator(), media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            assistant_response = service.get_response(
                user_message, sessions[user_message.session_id]
            )
            sessions[user_message.session_id].append(
                {"role": "assistant", "content": assistant_response}
            )
            return {"response": assistant_response}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
