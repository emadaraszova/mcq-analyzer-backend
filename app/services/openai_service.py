from app.core.config import settings
from openai import OpenAI
from fastapi import HTTPException
from app.schemas.clinical_scenario import ClinicalScenario


class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def get_response(self, user_message, session):
        """
        Generate a response using the Chat Completions API via the OpenAI client.
        """
        try:
            response = self.client.chat.completions.create(
                model=user_message.model,
                messages=session,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

    def get_stream_response(self, user_message, session):
        """
        Generate a streaming response using the Chat Completions API.
        """
        try:
            openai_stream = self.client.chat.completions.create(
                model=user_message.model,
                messages=session,
                stream=True,  # Enable streaming
            )
            for event in openai_stream:
                # Access the "delta" attribute safely
                delta = event.choices[0].delta
                if hasattr(delta, "content"):
                    yield delta.content  # Stream the content
        except Exception as e:
            yield f"Error: {str(e)}"

    def extract_clinical_info(self, questions: str, number_of_questions: int) -> list[ClinicalScenario]:
        """
        Extract structured information from questions using GPT API.s
        """
        try:
            response = self.client.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"Extract structured information *only* from the clinical scenarios that are part of the questions provided below."
                        f"There are/is {number_of_questions} question(s) in total."
                        "For each clinical scenario, create a JSON object conforming to the provided schema; no key should be missing:\n"
                        "- gender: [male, female, or null]\n"
                        "- age: [integer or null]\n"
                        "- symptoms: [string or null]\n"
                        "- family background: [string or null]\n\n"
                        "If information for a key cannot be found, use `null` as its value. If a clinical scenario mentions a diagnosis but not symptoms, "
                        "set the symptoms key to `null`. If no clinical scenario exists for a question, include the question but set all keys to `null`.\n\n"},
                    {"role": "user", "content": "Extract the information (gender, age, symptoms, and family background) from the clinical scenarios that are part of the provided question(s): {sanitized_questions}."},
                ],
                response_format=list[ClinicalScenario],
            )

            response = completion.choices[0].message
            if math_response.parsed:
                return math_response.parsed
            elif math_response.refusal:
                return math_response.refusal
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GPT error: {str(e)}")