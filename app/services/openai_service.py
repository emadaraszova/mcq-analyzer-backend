from app.core.config import settings
from openai import OpenAI
from app.services.base_service import BaseService
from app.schemas.clinical_scenario import StructuredInfo
from textwrap import dedent
import json


class OpenAIService(BaseService):
    def __init__(self):
        """
        Initialize the OpenAIService with the API key.
        """
        super().__init__(api_key=settings.OPENAI_API_KEY)
        self.client = OpenAI(api_key=self.api_key)

    def get_response(self, user_message, session):
        """
        Generate a response using the Chat Completions API.
        """
        try:
            response = self.client.chat.completions.create(
                model=user_message.model,
                messages=session,
                temperature=0.2,
                top_p=1.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.handle_exception(e, "OpenAI")

    def get_stream_response(self, user_message, session):
        """
        Generate a streaming response using the Chat Completions API.
        """
        try:
            openai_stream = self.client.chat.completions.create(
                model=user_message.model,
                messages=session,
                stream=True,
                temperature=0,
                top_p=1.0,
            )
            for event in openai_stream:
                delta = event.choices[0].delta
                if hasattr(delta, "content"):
                    yield delta.content
        except Exception as e:
            yield f"Error: {str(e)}"

    def extract_clinical_info(self, questions: str, number_of_questions: int) -> StructuredInfo:
        """
        Extract structured information from questions using the GPT API.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": dedent(
                            """
                            You are a clinical data extractor. You will be provided with test questions,
                            and your goal will be to output structured information from the clinical scenarios 
                            within the questions. Each clinical scenario must conform to the specified JSON schema, 
                            including details such as gender, ethnicity, and age.
                            Use `null` for any missing information.
                            """
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"There are {number_of_questions} question(s). "
                            f"Extract structured information from the following clinical scenarios: {questions}."
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
                                    "description": "A list of clinical scenarios for structured information.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "gender": {
                                                "type": "string",
                                                "description": "The gender of the patient."
                                            },
                                            "ethnicity": {
                                                "type": "string",
                                                "description": "Ethnicity or race."
                                            },
                                            "age": {
                                                "type": "integer",
                                                "description": "The age of the patient."
                                            }
                                        },
                                        "required": ["gender", "ethnicity", "age"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["questions"],
                            "additionalProperties": False
                        }
                    }
                },
            )

            structured_data = response.choices[0].message.content
            if isinstance(structured_data, str):
                structured_data = json.loads(structured_data)

            return structured_data
        except Exception as e:
            self.handle_exception(e, "OpenAI")
