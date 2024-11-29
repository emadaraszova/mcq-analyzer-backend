from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv
import os
import requests



API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
API_URL = "https://api.openai.com/v1/chat/completions"


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development; restrict in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# In-memory storage for conversation history
sessions: Dict[str, List[Dict[str, str]]] = {}

# Define the request model
class UserMessage(BaseModel):
    session_id: str  # Unique session identifier
    message: str     # User's new message
    model: str  # Default model
    # temperature: float = 0.7  # Optional, default value
    # top_p: float = 1.0        # Optional, default value
    # frequency_penalty: float = 0.0  # Optional, default value
    # presence_penalty: float = 0.0   # Optional, default value
    

@app.post("/api/generate-response")
async def generate_response(user_message: UserMessage):
    try:
        if user_message.session_id not in sessions:
            sessions[user_message.session_id] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

        sessions[user_message.session_id].append(
            {"role": "user", "content": user_message.message}
        )

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": user_message.model,
            "messages": sessions[user_message.session_id],
        }

        # Send the request to OpenAI's API
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()

        # Log the entire response for debugging
        data = response.json()
        print("OpenAI API Response:", data)  # Log the raw API response

        # Extract the assistant's response
        assistant_response = data["choices"][0]["message"]["content"]

        # Append assistant's response to the conversation
        sessions[user_message.session_id].append(
            {"role": "assistant", "content": assistant_response}
        )

        # Return only the assistant's response
        return {"response": assistant_response}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except KeyError as e:
        print("KeyError in response structure:", e)  # Debug unexpected structure
        raise HTTPException(status_code=500, detail="Unexpected response structure.")
