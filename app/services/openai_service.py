from openai import OpenAI
from app.core.config import settings
from fastapi import HTTPException
from starlette.responses import StreamingResponse



class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def get_response(self, user_message, session, stream: bool):
        """
        Generate a response using the Chat Completions API via the OpenAI client.

        Args:
            user_message: UserMessage object containing the user's input and model.
            session: List of conversation messages (with roles and content).

        Returns:
            str: The assistant's response or a StreamingResponse.
        """
        try:
            if stream:
                response = self.client.chat.completions.create(
                    model=user_message.model,
                    messages=session,
                    stream=True,
                    # response_format={"type": "text"},  # Specify the response format
                    # temperature=0.7,         # Adjust randomness
                    # max_tokens=2048,         # Set max response length
                    # top_p=0.9,               # Use nucleus sampling
                    # frequency_penalty=0.0,   # Discourage word repetition
                    # presence_penalty=0.6     # Discourage topic repetition
                )

                # Stream the response chunks
                async def stream_response():
                    for chunk in response:
                        yield chunk.choices[0].delta.content

                return StreamingResponse(stream_response(), media_type="text/plain")
            else:
                # Generate a single response
                response = self.client.chat.completions.create(
                    model=user_message.model,
                    messages=session,
                )
                return response.choices[0].message.content

        except openai.APIConnectionError as e:
            raise HTTPException(
                status_code=503,
                detail="Unable to connect to the OpenAI API. Please try again later.",
            )

        except openai.RateLimitError as e:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait before making more requests.",
            )

        except openai.APIStatusError as e:
            error_message = f"OpenAI API error (status code: {e.status_code}): {e.response.get('error', {}).get('message', 'Unknown error')}"
            raise HTTPException(status_code=e.status_code, detail=error_message)

        except openai.APIError as e:
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred while processing the request: {str(e)}",
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An internal error occurred: {str(e)}",
            )
