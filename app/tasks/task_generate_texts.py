# app/tasks/task_generate_texts.py

import re
import uuid
from typing import Dict, Any, List
from app.services.openai_service import OpenAIService
from app.services.gemini_service import GeminiService


# helpers
SCENARIO_PATTERN = re.compile(r'XXX[\s\S]*?XXX', re.IGNORECASE)

def count_scenarios(text: str) -> int:
    return len(SCENARIO_PATTERN.findall(text))

def trim_to_n_questions(text: str, n: int) -> str:
    out_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        out_lines.append(line)
        if 'XXX' in line:
            if count_scenarios(''.join(out_lines)) >= n:
                break
    return ''.join(out_lines)

def select_service(m: str):
    if m.startswith("gemini"):
        return GeminiService()
    if m.startswith("gpt") or m == "chatgpt-4o-latest":
        return OpenAIService()
    raise ValueError(f"Invalid model specified: {m}. Supported: gemini*, gpt-*.")

def generate_texts(payload: Dict[str, Any]) -> Dict[str, Any]:
    message: str = payload.get("message", "")
    model: str = payload.get("model", "gpt-4o")
    target: int = int(payload.get("number_of_questions", 1))
    if not message or not message.strip():
        raise ValueError("Payload 'message' must be a non-empty string.")
    if target < 1:
        raise ValueError("'number_of_questions' must be >= 1.")

    session_id = str(uuid.uuid4())
    service = select_service(model)

    # Initial conversation
    session = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message},
    ]

    class _UserMessageShim:
        def __init__(self, message: str, model: str):
            self.message = message
            self.model = model

    user_msg = _UserMessageShim(message=message, model=model)

    accumulated_text = ""
    max_rounds = 8
    round_idx = 0

    # First round of generation
    first_reply = service.get_response(user_msg, session)
    accumulated_text = first_reply
    session.append({"role": "assistant", "content": first_reply})

    first_count = count_scenarios(accumulated_text)

    # If the first reply has 0 delimiters, return it immediately 
    if first_count == 0:
        return {
            "status": "completed",
            "session_id": session_id,
            "model": model,
            "message": message,
            "requested_number_of_questions": target,
            "actual_number_of_questions": 0,
            "response": accumulated_text,
            "note": "No XXX delimiters found in first iteration; returning initial response without follow-ups."
        }

    # If the first reply already meets/exceeds target, finish (trim if overshoot)
    if first_count >= target:
        if first_count > target:
            accumulated_text = trim_to_n_questions(accumulated_text, target)
        return {
            "status": "completed",
            "session_id": session_id,
            "model": model,
            "message": message,
            "requested_number_of_questions": target,
            "actual_number_of_questions": min(first_count, target),
            "response": accumulated_text,
        }

    # Subsequent rounds with  follow-up prompt to get more questions
    while round_idx < max_rounds:
        current_count = count_scenarios(accumulated_text)
        if current_count >= target:
            break

        remaining = target - current_count
        followup = (
            f"Please continue with the remaining {remaining} questions."
        )

        session.append({"role": "user", "content": followup})
        user_msg = _UserMessageShim(message=followup, model=model)

        assistant_text = service.get_response(user_msg, session)
        accumulated_text += ("\n\n" if accumulated_text else "") + assistant_text
        session.append({"role": "assistant", "content": assistant_text})

        round_idx += 1

    final_count = count_scenarios(accumulated_text)
    status = "completed" if final_count >= target else "partial"

    # If overshot, trim to exactly target
    if final_count > target:
        accumulated_text = trim_to_n_questions(accumulated_text, target)
        final_count = target

    return {
        "status": status,
        "session_id": session_id,
        "model": model,
        "message": message,
        "requested_number_of_questions": target,
        "actual_number_of_questions": final_count,
        "response": accumulated_text,
        "note": "Multi-round conversation used to reach target",
        }
