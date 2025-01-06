from app.core.config import settings
from openai import OpenAI
from app.services.base_service import BaseService
from app.schemas.clinical_scenario import StructuredInfo
from textwrap import dedent


class OpenAIService(BaseService):
    def __init__(self):
        """
        Initialize the OpenAIService with the API key.
        """
        super().__init__(api_key=settings.OPENAI_API_KEY)
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def get_response(self, user_message, session):
        """
        Generate a response using the Chat Completions API.
        """
        try:
            response = self.client.chat.completions.create(
                model=user_message.model,
                messages=session,
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
            )
            for event in openai_stream:
                delta = event.choices[0].delta
                if hasattr(delta, "content"):
                    yield delta.content  
        except Exception as e:
            yield f"Error: {str(e)}"

    from app.core.config import settings
from openai import OpenAI
from app.services.base_service import BaseService
from app.schemas.clinical_scenario import StructuredInfo
from textwrap import dedent


class OpenAIService(BaseService):
    def __init__(self):
        """
        Initialize the OpenAIService with the API key.
        """
        super().__init__(api_key=settings.OPENAI_API_KEY)
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def get_response(self, user_message, session):
        """
        Generate a response using the Chat Completions API.
        """
        try:
            response = self.client.chat.completions.create(
                model=user_message.model,
                messages=session,
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
            )
            for event in openai_stream:
                delta = event.choices[0].delta
                if hasattr(delta, "content"):
                    yield delta.content
        except Exception as e:
            yield f"Error: {str(e)}"

    def extract_clinical_info(self, questions: str, number_of_questions: int) -> StructuredInfo:
        """
        Extract structured information from questions using GPT API.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": dedent('''
                            You are a clinical data extractor. You will be provided with test questions,
                            and your goal will be to output structured information from the clinical scenarios 
                            within the questions. Each clinical scenario must conform to the specified JSON schema, 
                            including details such as gender, age, symptoms, and family background.
                            Use `null` for any missing information.
                        '''),
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
                                                "description": "The gender of the patient.",
                                            },
                                            "age": {
                                                "type": "string",
                                                "description": "The age of the patient.",
                                            },
                                            "symptoms": {
                                                "type": "string",
                                                "description": "The symptoms.",
                                            },
                                            "family_background": {
                                                "type": "string",
                                                "description": "Having similar issues in the family.",
                                            },
                                        },
                                        "required": [
                                            "gender",
                                            "age",
                                            "symptoms",
                                            "family_background",
                                        ],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["questions"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
            )

            structured_data = response.choices[0].message.content
            if isinstance(structured_data, str):
                import json
                structured_data = json.loads(structured_data)

            return structured_data
        except Exception as e:
            self.handle_exception(e, "OpenAI")
