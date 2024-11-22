from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv
import os
import requests
import google.generativeai as genai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("API keys for OpenAI and Gemini are not set.")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, List[Dict[str, str]]] = {}

class UserMessage(BaseModel):
    session_id: str
    message: str
    model: str

@app.post("/api/generate-response")
async def generate_response(user_message: UserMessage):
    try:
        
        if user_message.session_id not in sessions:
            sessions[user_message.session_id] = [
                {"role": "user", "content": "You are a helpful assistant."}
            ]

    
        sessions[user_message.session_id].append(
            {"role": "user", "content": user_message.message}
        )

        if user_message.model.startswith("gemini"):
            
            gemini_history = [
                {
                    "role": entry["role"],
                    "parts": [entry["content"]]  
                }
                for entry in sessions[user_message.session_id]
                if entry["role"] in ["user", "assistant"]
            ]

            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
            gemini_model = genai.GenerativeModel(
                model_name=user_message.model, generation_config=generation_config
            )
            chat_session = gemini_model.start_chat(history=gemini_history)
            response = chat_session.send_message(user_message.message)
            assistant_response = response.text

        else:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": user_message.model,
                "messages": sessions[user_message.session_id],
            }

            openai_response = requests.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
            )
            openai_response.raise_for_status()
            data = openai_response.json()
            assistant_response = data["choices"][0]["message"]["content"]

        sessions[user_message.session_id].append(
            {"role": "assistant", "content": assistant_response}
        )

        return {"response": assistant_response}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
