from fastapi import HTTPException

class BaseService:
    def __init__(self, api_key: str):
        """
        Base class for API services.
        """
    
        self.api_key = api_key

    def sanitize_input(self, input_text: str) -> str:
        """
        Sanitize the input text to ensure it's JSON-compatible.
        """
        import json
        return json.dumps(input_text)

    def handle_exception(self, e: Exception, service_name: str):
        """
        Handle exception and raise HTTP errors with consistent formatting.
        """
        raise HTTPException(status_code=500, detail=f"{service_name} error: {str(e)}")

    def normalize_output(self, questions: list) -> list:
        """
        Normalize clinical scenario data to ensure consistency in structure and verbosity.
        """
        normalized_questions = []
        for question in questions:
            normalized_questions.append({
                "gender": question.get("gender", "null"),
                "age": question.get("age", "null"),
                "ethinicity": question.get("ethinicity", "").strip().lower(),
            })
        return normalized_questions