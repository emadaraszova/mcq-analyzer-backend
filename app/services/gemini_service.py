"""Gemini service adapter using google-generativeai."""

from __future__ import annotations

import json  # used to parse JSON string responses
from typing import Any, Dict, List

import google.generativeai as genai

from app.core.config import settings
from app.schemas.clinical_analysis import StructuredInfo
from app.services.base_service import BaseService


class GeminiService(BaseService):
    """Service wrapper for the Gemini API."""

    def __init__(self) -> None:
        """Initialize Gemini with the configured API key."""
        super().__init__(api_key=settings.GEMINI_API_KEY)
        genai.configure(api_key=self.api_key)

    def _to_gemini_history(self, session: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert a generic chat session into Gemini's history format.

        Args:
            session: List of chat messages, each a dict with ``role`` and ``content``.

        Returns:
            A list of dicts in the format expected by Gemini:
            ``{"role": "user"|"model", "parts": [content]}``.
        """
        mapped: List[Dict[str, Any]] = []
        for entry in session:
            role = entry.get("role")
            content = entry.get("content", "")
            if role == "user":
                mapped.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                mapped.append({"role": "model", "parts": [content]})
        return mapped

    def get_response(self, user_message, session: List[Dict[str, Any]]) -> str:
        """Generate a single assistant reply (non-streaming).

        Args:
            user_message: Object with ``model`` and ``message`` attributes.
            session: Prior chat messages.

        Returns:
            Assistant text response.
        """
        gemini_history = self._to_gemini_history(session)

        generation_config = {
            # "temperature": 0.2,
            # "top_p": 1.0,
            # "top_k": 40,
            # "max_output_tokens": 8192,
        }

        try:
            gemini_model = genai.GenerativeModel(
                model_name=user_message.model,
                generation_config=generation_config,
            )
            chat_session = gemini_model.start_chat(history=gemini_history)
            response = chat_session.send_message(user_message.message)
            return response.text
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.handle_exception(e, "Gemini")

    def extract_clinical_info(
        self, questions: str, number_of_questions: int
    ) -> StructuredInfo:
        """Extract structured demographic fields from scenario text via Gemini.

        The model is instructed to return JSON conforming to :class:`StructuredInfo`.

        Args:
            questions: Raw clinical scenario text.
            number_of_questions: Count hint for the prompt.

        Returns:
            A parsed Python object matching the ``StructuredInfo`` schema.
        """
        try:
            # Sanitize clinical scenarios (ensures JSON-safe content).
            sanitized_questions = self.sanitize_input(questions)
            print("sant. questions:", sanitized_questions)  # debug aid

            gemini_model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=(
                    """
                    You are a clinical data extractor.
                    You will be provided with one or more clinical scenarios as plain text.
                    
                    For each scenario, extract and return the following structured fields:
                    - gender
                    - ethnicity
                    - age

                    The output must strictly follow the provided JSON schema,
                    returning an object with a top-level "questions" array.
                    Each item in the array must represent one scenario.

                    Use `null` for any missing information.
                    Do not add extra fields or commentary outside the JSON structure.
                    """
                ),
            )

            response = gemini_model.generate_content(
                f"There are {number_of_questions} scnario(s). "
                f"Extract structured information from the following clinical scenarios: {questions}.",
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=StructuredInfo,
                ),
            )

            # Provider returns JSON as text; parse to Python object.
            structured_data = response.text
            if isinstance(structured_data, str):
                structured_data = json.loads(structured_data)

            return structured_data
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.handle_exception(e, "Gemini")
