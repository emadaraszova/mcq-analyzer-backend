"""Base service utilities for provider integrations and normalization."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import HTTPException


class BaseService:
    """Base class for API-backed services.

    Attributes:
        api_key: Provider API key used for outbound requests.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize the service.

        Args:
            api_key: Provider API key.
        """
        self.api_key = api_key

    def sanitize_input(self, input_text: str) -> str:
        """Return a JSON-encoded version of the input text.

        This escapes control characters and quotes so the string can be safely
        embedded in JSON payloads.

        Note:
            The return value is a **JSON string literal**, including quotes.
            Example: ``hello"`` -> ``"hello\""``.

        Args:
            input_text: Raw text to encode.

        Returns:
            A JSON-encoded string literal.
        """
        return json.dumps(input_text)

    def handle_exception(self, e: Exception, service_name: str) -> None:
        """Raise a standardized HTTP 500 for provider errors.

        Args:
            e: The underlying exception.
            service_name: Human-readable provider name to surface in the error.

        Raises:
            HTTPException: Always raised with status 500 and a concise message.
        """
        raise HTTPException(status_code=500, detail=f"{service_name} error: {e}")

    def normalize_output(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize clinical scenario items to a consistent structure.

        Ensures expected keys exist and values are sanitized (trimmed/lower-cased
        where appropriate). Missing fields are filled with the string ``"null"``.

        Args:
            questions: A list of question/scenario dictionaries returned by a provider.

        Returns:
            A list of normalized dictionaries with at least:
            - ``sex`` (str)
            - ``ethnicity`` (str)
        """
        normalized: List[Dict[str, Any]] = []
        for q in questions:
            sex = (q.get("sex") or "null").strip()
            ethnicity_raw = q.get("ethnicity") or ""
            ethnicity = ethnicity_raw.strip().lower() or "null"

            normalized.append(
                {
                    "sex": sex,
                    "ethnicity": ethnicity,
                }
            )
        return normalized
