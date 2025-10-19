from app.core.config import settings
import google.generativeai as genai
from app.schemas.clinical_scenario import StructuredInfo
from app.services.base_service import BaseService



class GeminiService(BaseService):
    def __init__(self):
        """
        Initialize GeminiService with the API key.
        """
        super().__init__(api_key=settings.GEMINI_API_KEY)
        genai.configure(api_key=self.api_key)

    def _to_gemini_history(self, session):
        mapped = []
        for entry in session:
            role = entry.get("role")
            content = entry.get("content", "")
            if role == "user":
                mapped.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                mapped.append({"role": "model", "parts": [content]})
        return mapped


    def get_response(self, user_message, session):
        """
        Generate a response using the Gemini API without streaming.
        """
        gemini_history = self._to_gemini_history(session)

        generation_config = {
            "temperature": 0.2,
            "top_p": 1.0,
           # "top_k": 40,
           # "max_output_tokens": 8192,
        }

        try:
            gemini_model = genai.GenerativeModel(
                model_name=user_message.model,
                generation_config=generation_config
            )
            chat_session = gemini_model.start_chat(history=gemini_history)
            response = chat_session.send_message(user_message.message)
            return response.text
        except Exception as e:
            self.handle_exception(e, "Gemini")


    def extract_clinical_info(self, questions: str, number_of_questions: int) -> StructuredInfo:
        """
        Extract structured information from questions using Gemini API.
        """
        try:
            # Sanitize clinical scenarios
            sanitized_questions = self.sanitize_input(questions)
            print("sant. questions:", sanitized_questions)
            gemini_model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=(
                    f"Extract structured information from the clinical scenarios provided below."
                    f"There are/is {number_of_questions} scenatio(s) in total."
                    "For each clinical scenario, create a JSON object conforming to the provided schema:\n"
                    "- gender: [male, female, or null]\n"
                    "- ethnicity: [string or null]\n"
                    "If information for a key cannot be found, use `null` as its value."
                    "If no such information exists for a scenario, include the scenario but set all keys to `null`.\n\n"
                    "The output must be a JSON array where each element corresponds to one question."
                )
            )

            response = gemini_model.generate_content(
                f"Extract the information (gender and ethnicity) from the clinical scenarios that are part of the provided question(s): {sanitized_questions}",
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=StructuredInfo
                ),
            )

         
             # Extract the structured data from the response
            structured_data = response.text  # Adjust based on actual response format
            if isinstance(structured_data, str):
                import json
                structured_data = json.loads(structured_data)

            return structured_data
        except Exception as e:
            self.handle_exception(e, "Gemini")