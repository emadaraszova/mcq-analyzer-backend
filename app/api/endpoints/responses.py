from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.user_message import UserMessage
from app.services.openai_service import OpenAIService
from app.services.gemini_service import GeminiService

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
        # Ensure session exists
        session_id = user_message.session_id
        if session_id not in sessions:
            sessions[session_id] = [{"role": "system", "content": "You are a helpful assistant."}]

        # Add user message to session
        sessions[session_id].append({"role": "user", "content": user_message.message})

        # Select service based on model prefix
        service = select_service(user_message.model)

        # Generate response
        if stream:
            return generate_streaming_response(service, user_message, sessions[session_id])
        else:
            return generate_non_streaming_response(service, user_message, sessions[session_id])

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


def select_service(model: str):
    """
    Select the appropriate service based on the model prefix.
    """
    if model.startswith("gemini"):
        return GeminiService()
    elif model.startswith("gpt-") or model == "chatgpt-4o-latest":
        return OpenAIService()
    else:
        raise ValueError(f"Invalid model specified: {model}. Supported models: gemini, gpt-*.")


def generate_streaming_response(service, user_message: UserMessage, session):
    """
    Generate a streaming response using the service.
    """
    def response_generator():
        try:
            for chunk in service.get_stream_response(user_message, session):
                if chunk:
                    print(f"Yielding chunk: {chunk}")
                    yield f"data: {chunk}\n\n"
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: [Error] {str(e)}\n\n"

    return StreamingResponse(response_generator(), media_type="text/event-stream")


def generate_non_streaming_response(service, user_message: UserMessage, session):
    """
    Generate a non-streaming response using the service.
    """
    try:
        assistant_response = service.get_response(user_message, session)
        session.append({"role": "assistant", "content": assistant_response})
        return {"response": assistant_response}
    except Exception as e:
        print(f"Non-streaming error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response.")
