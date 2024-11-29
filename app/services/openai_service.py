from openai import OpenAI
from app.core.config import settings
from fastapi import HTTPException

class OpenAIService:
    def __init__(self):
       
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def get_response(self, user_message, session):
        """
        Generate a response using the Chat Completions API via the OpenAI client.

        Args:
            user_message: UserMessage object containing the user's input and model.
            session: List of conversation messages (with roles and content).

        Returns:
            str: The assistant's response.
        """
        try:
            # Use the new client interface for chat completions
            response = self.client.chat.completions.create(
                model=user_message.model,  # e.g., "gpt-4o" or "gpt-4o-mini"
                messages=session,
                # response_format={"type": "text"},  # Specify the response format
                # temperature=0.7,         # Adjust randomness
                # max_tokens=2048,         # Set max response length
                # top_p=0.9,               # Use nucleus sampling
                # frequency_penalty=0.0,   # Discourage word repetition
                # presence_penalty=0.6     # Discourage topic repetition
            )

            # Extract the response text
            return response.choices[0].message.content  

        # Handle connection-related errors
        except openai.APIConnectionError as e:
            raise HTTPException(
                status_code=503, 
                detail="Unable to connect to the OpenAI API. Please try again later."
            )

        # Handle rate-limiting errors
        except openai.RateLimitError as e:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait before making more requests."
            )

        # Handle specific API status errors
        except openai.APIStatusError as e:
            # Provide additional context for the error
            error_message = f"OpenAI API error (status code: {e.status_code}): {e.response.get('error', {}).get('message', 'Unknown error')}"
            raise HTTPException(status_code=e.status_code, detail=error_message)

        # Handle other API-related errors
        except openai.APIError as e:
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred while processing the request: {str(e)}"
            )

        # Handle general unexpected exceptions
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An internal error occurred: {str(e)}"
            )