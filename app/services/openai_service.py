"""OpenAI service adapter using Chat Completions."""

from __future__ import annotations

import json
from textwrap import dedent

from openai import OpenAI

from app.core.config import settings
from app.schemas.clinical_analysis import StructuredInfo
from app.services.base_service import BaseService


class OpenAIService(BaseService):
    """Service wrapper for OpenAI's Chat Completions API.

    Inherits common helpers (e.g., error handling) from :class:`BaseService`.
    """

    def __init__(self) -> None:
        """Initialize the OpenAI client with the API key."""
        super().__init__(api_key=settings.OPENAI_API_KEY)
        self.client = OpenAI(api_key=self.api_key)

    def get_response(self, user_message, session) -> str:
        """Generate a single assistant reply.

        Args:
            user_message: Object with at least a ``model`` attribute.
            session: Chat history as a list of ``{role, content}`` dicts.

        Returns:
            Assistant message content as a string.

        Notes:
            Any provider exceptions are passed through :meth:`BaseService.handle_exception`.
        """
        try:
            response = self.client.chat.completions.create(
                model=user_message.model,
                messages=session,
                # temperature=0.2,
                # top_p=1.0,
            )
            return response.choices[0].message.content
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.handle_exception(e, "OpenAI")

    def extract_clinical_info(
        self, questions: str, number_of_questions: int
    ) -> StructuredInfo:
        """Extract structured clinical information from scenario text.

        The model is instructed to return a JSON object with a top-level
        ``questions`` array. Each element contains ``sex``, ``ethnicity``,
        and ``age`` fields.

        Args:
            questions: Raw scenario text (one or more scenarios).
            number_of_questions: Count hint for the prompt.

        Returns:
            Parsed JSON (as a Python dict) conforming to :class:`StructuredInfo`.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
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
                                        "A list of clinical scenarios for structured information."
                                    ),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "sex": {
                                                "type": "string",
                                                "description": "The sex of the patient.",
                                            },
                                            "ethnicity": {
                                                "type": "string",
                                                "description": "Ethnicity or race.",
                                            },
                                            "age": {
                                                "type": "integer",
                                                "description": "The age of the patient.",
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
            self.handle_exception(e, "OpenAI")
