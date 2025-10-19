from app.core.config import settings
from openai import OpenAI
from app.schemas.clinical_scenario import StructuredInfo
from app.services.base_service import BaseService
from textwrap import dedent
import json


class EInfraService(BaseService):
    def __init__(self):
        """
        Initialize the EInfraService with the API key and base URL.
        """
        super().__init__(api_key=settings.E_INFRA_API_KEY)
        self.client = OpenAI(api_key=self.api_key, base_url=getattr(settings, "E_INFRA_BASE_URL", "https://chat.ai.e-infra.cz/api"))

    def get_response(self, user_message, session):
        """
        Generate a response using Chat Completions API (OpenAI compatible).
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
            self.handle_exception(e, "E-INFRA")

    def extract_clinical_info(self, questions: str, number_of_questions: int) -> StructuredInfo:
        """
        Use E-INFRA model to extract structured information from questions.
        """
        try:
            response = self.client.chat.completions.create(
                model="llama3.3:latest",
                messages=[
                    {
                        "role": "system",
                        "content": dedent(
                            """
                            You are a clinical data extractor. You will be provided with clinical scenario(s),
                            and your goal will be to output structured information from the clinical scenarios.
                            Structured information for each clinical scenario must conform to the specified JSON schema, 
                            including details such as gender, ethnicity, and age.
                            Use `null` for any missing information.
                            """
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"There are {number_of_questions} scnario(s). "
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
            self.handle_exception(e, "E-INFRA")