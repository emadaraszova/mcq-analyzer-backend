"""E-INFRA service adapter (OpenAI-compatible Chat Completions)."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, List, Dict

from openai import OpenAI

from app.core.config import settings
from app.schemas.clinical_analysis import StructuredInfo
from app.services.base_service import BaseService


class EInfraService(BaseService):
    """Service wrapper for the E-INFRA endpoint.

    Uses an OpenAI-compatible Chat Completions API exposed by E-INFRA.
    Inherits common helpers (error handling, etc.) from :class:`BaseService`.
    """

    def __init__(self) -> None:
        """Initialize client with API key and base URL."""
        super().__init__(api_key=settings.E_INFRA_API_KEY)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=getattr(
                settings, "E_INFRA_BASE_URL", "https://chat.ai.e-infra.cz/api"
            ),
        )

    def get_response(self, user_message: Any, session: List[Dict[str, str]]) -> str:
        """Generate a single assistant reply.

        Args:
            user_message: Object with at least ``model`` attribute used by the API.
            session: Full chat history as a list of ``{role, content}`` dicts.

        Returns:
            The assistant message content as a string.

        Raises:
            HTTPException: Wrapped by :meth:`BaseService.handle_exception` on error.
        """
        try:
            response = self.client.chat.completions.create(
                model=user_message.model,
                messages=session,
                # temperature=0.2,
                # top_p=1.0,
            )
            return response.choices[0].message.content
        except Exception as e:  # noqa: BLE001 - surface provider error consistently
            self.handle_exception(e, "E-INFRA")
            raise  # for type checkers; handle_exception always raises

    def extract_clinical_info(
        self, questions: str, number_of_questions: int
    ) -> StructuredInfo:
        """Extract structured clinical information from scenario text.

        The model is instructed to return a JSON object conforming to a schema
        that includes a top-level ``questions`` array. Each element contains
        ``sex``, ``ethnicity``, and ``age`` fields.

        Args:
            questions: Raw scenario text (1 or more scenarios).
            number_of_questions: Count hint for the prompt.

        Returns:
            Parsed JSON object as a Python dict matching ``StructuredInfo``.

        Raises:
            HTTPException: Wrapped by :meth:`BaseService.handle_exception` on error.
        """
        try:
            response = self.client.chat.completions.create(
                model="llama3.3:latest",
                messages=[
                    {
                        "role": "system",
                        "content": dedent(
                            """
                            You are a clinical data extractor.
                            You will be provided with one or more clinical scenarios as plain text.
                            
                            For each scenario, extract and return the following structured fields:
                            - sex
                            - ethnicity
                            - age

                            The output must strictly follow the provided JSON schema,
                            returning an object with a top-level "questions" array.
                            Each item in the array must represent one scenario.

                            Use `null` for any missing information.
                            Do not add extra fields or commentary outside the JSON structure.
                            """
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"There are {number_of_questions} scnario(s). "
                            "Extract structured information from the following "
                            f"clinical scenarios: {questions}."
                        ),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_info",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "questions": {
                                    "type": "array",
                                    "description": (
                                        "A list of structured demographic information objects."
                                    ),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "sex": {
                                                "type": "string",
                                                "description": (
                                                    "The sex of the patient."
                                                ),
                                            },
                                            "ethnicity": {
                                                "type": "string",
                                                "description": "Ethnicity or race.",
                                            },
                                            "age": {
                                                "type": "integer",
                                                "description": (
                                                    "The age of the patient."
                                                ),
                                            },
                                        },
                                        "required": ["sex", "ethnicity", "age"],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["questions"],
                            "additionalProperties": False,
                        },
                    },
                },
            )

            # Provider returns JSON as a string; parse to Python object.
            structured_data = response.choices[0].message.content
            if isinstance(structured_data, str):
                structured_data = json.loads(structured_data)

            return structured_data
        except Exception as e:  # noqa: BLE001
            self.handle_exception(e, "E-INFRA")
            raise  # for type checkers; handle_exception always raises
