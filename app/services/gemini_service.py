from app.core.config import settings
import google.generativeai as genai
from fastapi import HTTPException
from app.schemas.clinical_scenario import ClinicalScenario


class GeminiService:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)

    def get_response(self, user_message, session):
        """
        Generate a response using the Gemini API without streaming.
        """
        gemini_history = [
            {
                "role": entry["role"],
                "parts": [entry["content"]]
            }
            for entry in session
            if entry["role"] in ["user", "assistant"]
        ]

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
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
            raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    def get_stream_response(self, user_message, session):
        """
        Generate a streaming response using the Gemini API.
        """
        gemini_history = [
            {
                "role": entry["role"],
                "parts": [entry["content"]]
            }
            for entry in session
            if entry["role"] in ["user", "assistant"]
        ]

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        try:
            gemini_model = genai.GenerativeModel(
                model_name=user_message.model,
                generation_config=generation_config
            )

            chat_session = gemini_model.start_chat(history=gemini_history)
            response_stream = chat_session.send_message(
                user_message.message,
                stream=True
            )
            for chunk in response_stream:
                # Only yield chunks with text content
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error: {str(e)}"

    def sanitize_input(self, input_text: str) -> str:
        """
        Sanitize the input text to ensure it's JSON-compatible.
        """
        import json
        # Escape special characters
        return json.dumps(input_text)

    def extract_clinical_info(self, clinical_scenarios: str, number_of_questions: int) -> list[ClinicalScenario]:
        """
        Extract structured information from a clinical scenario using Gemini API.
        """
        try:
            # Sanitize clinical scenarios
            sanitized_scenarios = self.sanitize_input(clinical_scenarios)
            gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=(
                    f"Extract structured information *only* from the clinical scenarios that are part of the questions provided below."
                    f"There are/is {number_of_questions} question(s) in total."
                    "For each clinical scenario, create a JSON object conforming to the provided schema:\n"
                    "- gender: [male, female, or null]\n"
                    "- age: [integer or null]\n"
                    "- symptoms: [string or null]\n"
                    "- family background: [string or null]\n\n"
                    "If information for a key cannot be found, use `null` as its value. If a clinical scenario mentions a diagnosis but not symptoms, "
                    "set the symptoms key to `null`. If no clinical scenario exists for a question, include the question but set all keys to `null`.\n\n"
                    "The output must be a JSON array where each element corresponds to one question."
                )
            )
            print("these are clinical scenarios:", sanitized_scenarios)

            response = gemini_model.generate_content(
                f"Extract the information (age, gender, symptoms, and family background) from the clinical scenarios that are part of the provided question(s): {sanitized_scenarios}",
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=list[ClinicalScenario]
                ),
            )

            # Extract the structured data from the response
            structured_data = response.text  # Adjust based on actual response format
            if isinstance(structured_data, str):
                import json
                structured_data = json.loads(structured_data)

            return structured_data  # Ensure this is a list or dict
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")
